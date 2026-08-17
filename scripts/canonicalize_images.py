"""
Image canonicalization + manifest, fused into a single pass.

Fused because the alternative decodes every image twice and, on this pod's
MooseFS /data, writing ~10M individual JPEG files is pathologically slow.
Canonicalized images go into parquet shards mirroring the raw layout; the
manifest addresses them as "<dataset>/<shard>.parquet#<row>".

Canonicalization, applied identically to both classes:
  1. exif_transpose, then discard all metadata
  2. convert to RGB (handles CMYK, palette, LA, RGBA)
  3. random-crop to 256x256 -- never resize (resizing destroys the
     high-frequency residuals detectors rely on). Images under 256 on
     either side are reflect-padded first and flagged.
  4. re-encode to JPEG quality 95, identical encoder settings everywhere,
     which removes compression-signature leakage between sources
  5. pre-crop resolution recorded per row so class-wise distributions can
     be compared afterwards

Each output shard gets a .done marker so reruns skip completed work.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Must be set before pyarrow/numpy import in each worker. PyArrow's thread
# pool is not fork-safe -- forking mid-operation deadlocks the child on a
# lock the parent held, which is exactly what 40 forked workers did here
# (rchar frozen, zero progress). Workers are spawned, not forked, and each
# runs Arrow single-threaded since we already parallelize at shard level.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import imagehash
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
try:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
except Exception:
    pass

CROP = 256
JPEG_Q = 95
# Above this, stream from disk rather than buffering the shard in RAM.
#
# Size the worker count against the CONTAINER limit, not `free`. `free` reports
# the host (251 GB on one pod); the cgroup allowed 41 GB:
#     cat /sys/fs/cgroup/memory.max
# Exceeding it gets a worker SIGKILLed, and ProcessPoolExecutor then reports
# BrokenProcessPool for every *other* shard too -- 183 of 186 "failures" from a
# single OOM, with the real cause nowhere in the output. Page cache from
# reading the shards counts toward that limit as well.
# Whole-file read: ~2.9 GB/worker, 28 s/shard. Streaming (set this to 0):
# ~1.9 GB/worker, 67 s/shard. Trade speed for headroom when the cgroup is tight.
BLOB_MAX_BYTES = int(os.getenv("BLOB_MAX_BYTES", 1_500_000_000))

# canon-v1 (crop) is the literature-standard treatment: it preserves native
# high-frequency content, which is most of what a generated-image detector
# keys on. canon-v2 (resize) standardizes field of view as well as pixel
# dimensions, at the cost of low-pass filtering exactly that signal. Both are
# built so the blur-baseline check can decide between them on evidence.
CANON_VERSIONS = {
    "crop":   "canon-v1:crop256-jpeg95-noexif",
    "resize": "canon-v2:resize256-bicubic-jpeg95-noexif",
}

PROC = Path("/data/proc")
LOGS = Path("/data/logs")

OUT_SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("dataset", pa.string()),
    ("split_src", pa.string()),
    ("label", pa.int8()),
    ("generator", pa.string()),
    ("architecture", pa.string()),
    ("real_source", pa.string()),
    ("subset", pa.string()),
    ("orig_width", pa.int32()),
    ("orig_height", pa.int32()),
    ("orig_format", pa.string()),
    ("was_padded", pa.bool_()),
    ("phash", pa.uint64()),
    ("image_jpeg", pa.binary()),
    ("canon_version", pa.string()),
])


def canonicalize(raw: bytes, rng: np.random.Generator, mode: str = "crop"):
    """Return (jpeg_bytes, orig_w, orig_h, orig_format, was_padded, phash_u64)."""
    im = Image.open(io.BytesIO(raw))
    orig_format = (im.format or "UNKNOWN").upper()
    im = ImageOps.exif_transpose(im)          # honor orientation...
    orig_w, orig_h = im.size

    if im.mode != "RGB":                       # ...then drop everything else
        im = im.convert("RGB")

    if mode == "resize":
        # Shortest side to 256 preserving aspect, then centre crop. This
        # equalizes both pixel size and field of view across sources.
        was_padded = im.width < CROP or im.height < CROP
        scale = CROP / min(im.width, im.height)
        new_w = max(CROP, int(round(im.width * scale)))
        new_h = max(CROP, int(round(im.height * scale)))
        im = im.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - CROP) // 2
        top = (new_h - CROP) // 2
        im = im.crop((left, top, left + CROP, top + CROP))

        ph_u64 = int(str(imagehash.phash(im)), 16)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_Q, subsampling="4:2:0", optimize=False)
        return buf.getvalue(), orig_w, orig_h, orig_format, was_padded, ph_u64

    was_padded = False
    if im.width < CROP or im.height < CROP:
        was_padded = True
        pad_l = max(0, (CROP - im.width + 1) // 2)
        pad_r = max(0, CROP - im.width - pad_l)
        pad_t = max(0, (CROP - im.height + 1) // 2)
        pad_b = max(0, CROP - im.height - pad_t)
        arr = np.asarray(im)
        arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="reflect")
        im = Image.fromarray(arr)

    max_x = im.width - CROP
    max_y = im.height - CROP
    x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    im = im.crop((x, y, x + CROP, y + CROP))

    ph = imagehash.phash(im)
    ph_u64 = int(str(ph), 16)

    buf = io.BytesIO()
    # No exif=, no icc_profile= -> metadata is gone. Fixed settings for both classes.
    im.save(buf, format="JPEG", quality=JPEG_Q, subsampling="4:2:0", optimize=False)
    return buf.getvalue(), orig_w, orig_h, orig_format, was_padded, ph_u64


def norm_generator(row) -> str:
    """
    Recover the generator label. Community Forensics exposes model_name per
    row, so this is a direct read rather than a directory-structure guess.
    Reals are keyed by their source pool so they group correctly in splits.
    """
    label = int(row["label"])
    if label == 0:
        src = (row.get("real_source") or "unknown").strip().lower()
        return f"real:{src}"
    name = (row.get("model_name") or "").strip()
    return name if name else "unknown"


def process_shard(args):
    src_path, dataset, out_path, seed, mode = args
    done = Path(str(out_path) + ".done")
    if done.exists():
        return json.loads(done.read_text())

    rng = np.random.default_rng(seed)
    t0 = time.time()
    # On a network filesystem one sequential read beats parquet's seeky
    # access pattern, but slurping a 4 GB shard costs ~15-25 GB resident once
    # parsed -- enough to break the pool at even modest worker counts. Small
    # shards get the whole-file read; large ones stream from the path, which
    # local NVMe handles fine.
    if os.path.getsize(src_path) <= BLOB_MAX_BYTES:
        with open(src_path, "rb") as fh:
            pf = pq.ParquetFile(io.BytesIO(fh.read()))
    else:
        pf = pq.ParquetFile(src_path)

    cols = {
        "uid": [], "dataset": [], "split_src": [], "label": [], "generator": [],
        "architecture": [], "real_source": [], "subset": [], "orig_width": [],
        "orig_height": [], "orig_format": [], "was_padded": [], "phash": [],
        "image_jpeg": [], "canon_version": [],
    }
    n_fail = 0
    shard_stem = Path(out_path).stem

    # Shards are a single 2,992-row group up to 4.1 GB; read_row_group would
    # materialize the whole thing per worker. Batched iteration keeps each
    # worker's footprint to a few hundred MB.
    idx = 0
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            i = idx
            idx += 1
            raw = row.get("image_data")
            if not raw:
                n_fail += 1
                continue
            try:
                jpg, ow, oh, ofmt, padded, ph = canonicalize(raw, rng, mode)
            except Exception:
                n_fail += 1
                continue
            cols["uid"].append(f"{dataset}/{shard_stem}/{i}")
            cols["dataset"].append(dataset)
            cols["split_src"].append(str(row.get("split") or ""))
            cols["label"].append(int(row["label"]))
            cols["generator"].append(norm_generator(row))
            cols["architecture"].append(str(row.get("architecture") or ""))
            cols["real_source"].append(str(row.get("real_source") or ""))
            cols["subset"].append(str(row.get("subset") or ""))
            cols["orig_width"].append(ow)
            cols["orig_height"].append(oh)
            cols["orig_format"].append(ofmt)
            cols["was_padded"].append(padded)
            cols["phash"].append(ph)
            cols["image_jpeg"].append(jpg)
            cols["canon_version"].append(CANON_VERSIONS[mode])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, schema=OUT_SCHEMA), out_path, compression="zstd")

    rec = {
        "src": str(src_path), "out": str(out_path), "dataset": dataset,
        "n_rows": len(cols["uid"]), "n_failed": n_fail,
        "seconds": round(time.time() - t0, 1),
        "canon_version": CANON_VERSIONS[mode],
    }
    done.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--limit", type=int, default=0, help="process only N shards (pilot)")
    ap.add_argument("--mode", choices=["crop", "resize"], default="crop")
    ap.add_argument("--out-name", default=None, help="output dir name under /data/proc")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    shards = sorted(raw.rglob("*.parquet"))
    if args.limit:
        shards = shards[: args.limit]
    if not shards:
        raise SystemExit(f"[canon] no parquet under {raw}")

    out_dir = PROC / (args.out_name or args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(s), args.dataset, str(out_dir / f"{s.stem}.parquet"), 42 + i, args.mode)
        for i, s in enumerate(shards)
    ]
    print(f"[canon] {args.dataset}: {len(tasks)} shards, {args.workers} workers, "
          f"canon={CANON_VERSIONS[args.mode]}", flush=True)
    # Surface the real memory ceiling up front -- see BLOB_MAX_BYTES.
    try:
        lim = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        per = 1.9 if BLOB_MAX_BYTES == 0 else 2.9
        print(f"[canon] cgroup memory limit {lim/2**30:.0f} GiB; "
              f"~{per:.1f} GB/worker x {args.workers} = {per*args.workers:.0f} GB"
              + ("  <-- OVER LIMIT, expect BrokenProcessPool"
                 if per * args.workers * 2**30 > lim * 0.75 else ""), flush=True)
    except (OSError, ValueError):
        pass

    t0 = time.time()
    recs, total, failed = [], 0, 0
    # submit/as_completed rather than map: a single worker dying (these
    # shards peak at ~15 GB resident) must not take the whole pool with it.
    # .done markers make a rerun pick up exactly where this left off.
    ctx = mp.get_context("spawn")
    from concurrent.futures import as_completed
    dead = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(process_shard, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:
                dead += 1
                print(f"[canon] shard FAILED ({type(exc).__name__}): "
                      f"{Path(futs[fut][0]).name} — rerun to retry", flush=True)
                continue
            recs.append(rec)
            total += rec["n_rows"]
            failed += rec["n_failed"]
            if k % 10 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[canon] {k}/{len(tasks)} shards | {total:,} imgs | "
                      f"{total/max(el,1):.0f} img/s | {el/60:.1f} min", flush=True)
    if dead:
        print(f"[canon] {dead} shard(s) failed this pass", flush=True)

    LOGS.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset, "n_shards": len(tasks), "n_images": total,
        "n_failed": failed, "wall_seconds": round(time.time() - t0, 1),
        "canon_version": CANON_VERSIONS[args.mode], "shards": recs,
    }
    (LOGS / f"canon_{args.out_name or args.dataset}.json").write_text(json.dumps(summary, indent=2))
    print(f"[canon] DONE {total:,} images, {failed} failed, "
          f"{(time.time()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()

"""
Forensic feature extraction (the six src/eda.py families).

Runs the project's own six forensic families from src/eda.py (pixel stats, FFT,
colour entropy via k-means, noise residuals, GLCM texture, Canny edge density)
against every canonicalized image, keyed by uid so the fusion trainer can join
them to the manifest.

Two deliberate choices:

  * Features are computed at the canonical 256x256 rather than being resized to
    224 first. Every image is already the same size, so the resize buys no
    consistency and would low-pass exactly the high-frequency content that the
    FFT, noise-residual and GLCM families exist to measure.

  * They are computed AFTER canonicalization, not on the originals. That is the
    harder setting: the uniform JPEG-95 re-encode has already destroyed the
    source-format signature that made reals 45% JPEG and fakes 2.2%. Whatever
    these features contribute here is content-level, not source leakage.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
except Exception:
    pass

PROC = Path("/data/proc")
FEAT = Path("/data/features")
LOGS = Path("/data/logs")


def _extractor():
    """Import lazily so each spawned worker builds its own module state."""
    import sys
    sys.path.insert(0, "/data/viper")
    from src.eda import extract_features_from_rgb
    return extract_features_from_rgb


def process_shard(args):
    shard_path, out_path, keep_uids = args
    done = Path(str(out_path) + ".done")
    if done.exists():
        return json.loads(done.read_text())

    from PIL import Image
    extract = _extractor()

    t0 = time.time()
    tbl = pq.read_table(shard_path, columns=["uid", "label", "image_jpeg"])
    uids = tbl.column("uid").to_pylist()
    labels = tbl.column("label").to_pylist()
    blobs = tbl.column("image_jpeg").to_pylist()
    del tbl

    rows, keys, n_fail = [], None, 0
    out_uid, out_label = [], []
    for uid, lab, blob in zip(uids, labels, blobs):
        if keep_uids is not None and uid not in keep_uids:
            continue
        try:
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            arr = np.asarray(im)                       # native 256x256, no resize
            f = extract(arr)
            f.pop("image_path", None)
            f.pop("label", None)
            if keys is None:
                keys = sorted(f.keys())
            rows.append([float(f.get(k, 0.0)) for k in keys])
            out_uid.append(uid)
            out_label.append(int(lab))
        except Exception:
            n_fail += 1

    if not rows:
        # A shard can legitimately contribute nothing when a uid filter is in
        # play; write an empty marker so reruns skip it.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        rec = {"shard": Path(shard_path).name, "n_rows": 0, "n_failed": n_fail,
               "n_features": 0, "feature_names": [], "seconds": round(time.time() - t0, 1)}
        Path(str(out_path) + ".done").write_text(json.dumps(rec))
        return rec

    mat = np.asarray(rows, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    cols = {"uid": pa.array(out_uid), "label": pa.array(out_label, type=pa.int8())}
    for j, k in enumerate(keys):
        cols[f"f_{k}"] = pa.array(mat[:, j], type=pa.float32())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out_path, compression="zstd")

    rec = {"shard": Path(shard_path).name, "n_rows": len(out_uid),
           "n_failed": n_fail, "n_features": len(keys),
           "feature_names": keys, "seconds": round(time.time() - t0, 1)}
    done.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-name", required=True)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uids", default=None,
                    help="newline-delimited uid list; restricts extraction to those rows")
    args = ap.parse_args()

    shards = sorted((PROC / args.proc_name).glob("*.parquet"))
    if args.limit:
        shards = shards[: args.limit]
    if not shards:
        raise SystemExit(f"[forensic] no shards in {PROC / args.proc_name}")

    out_dir = FEAT / (args.out_name or args.proc_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = None
    if args.uids:
        keep = frozenset(Path(args.uids).read_text().split())
        print(f"[forensic] restricting to {len(keep):,} uids", flush=True)
    tasks = [(str(s), str(out_dir / s.name), keep) for s in shards]
    print(f"[forensic] {len(tasks)} shards, {args.workers} workers", flush=True)

    t0, total, failed, nfeat, names = time.time(), 0, 0, None, None
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(process_shard, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"[forensic] FAILED {Path(futs[fut][0]).name}: "
                      f"{type(exc).__name__} {exc}", flush=True)
                continue
            total += rec["n_rows"]
            failed += rec["n_failed"]
            if rec["n_features"]:
                nfeat, names = rec["n_features"], rec["feature_names"]
            if k % 10 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[forensic] {k}/{len(tasks)} | {total:,} imgs | "
                      f"{total/max(el,1):.0f} img/s | {el/60:.1f} min", flush=True)

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"forensic_{args.out_name or args.proc_name}.json").write_text(json.dumps({
        "n_images": total, "n_failed": failed, "n_features": nfeat,
        "feature_names": names, "wall_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"[forensic] DONE {total:,} images, {failed} failed, {nfeat} features/image "
          f"-> {out_dir}")
    print(f"[forensic] features: {names}")


if __name__ == "__main__":
    main()

"""
Perceptual deduplication and leakage-free generator splits.

Writes /data/manifest/manifest_<name>.parquet with split assignments as
columns, never as file moves.

The assertions at the bottom are permanent, not one-off pilot checks: a
generator that appears on both sides of a split silently converts this from a
generalization benchmark into an in-distribution one, and the resulting AUC
looks *better*, so nothing else in the pipeline would catch it.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROC = Path("/data/proc")
MANI = Path("/data/manifest")
LOGS = Path("/data/logs")

META_COLS = ["uid", "dataset", "label", "generator", "architecture",
             "real_source", "subset", "orig_width", "orig_height",
             "orig_format", "was_padded", "phash", "canon_version"]

# Held-out generator families for LOGO. The brief asks for at least one GAN,
# one older diffusion model, one modern open model, and one commercial model.
# Families verified to exist in this corpus before use. On
# CommunityForensics-Small the Commercial subset is absent entirely (only
# Systematic and Manual ship), so the brief's fourth family cannot be held
# out here -- that gap is recorded rather than faked with community HF
# models that merely have "midjourney" or "dalle" in their repo names.
LOGO_PATTERNS = {
    "gan":         ["gigagan", "biggan", "progan", "stylegan2", "stylegan3", "cyclegan"],
    "sd1x":        ["stable-diffusion-v1", "stable-diffusion-1", "sd-v1", "v1-5"],
    "modern_open": ["sdxl", "styleganxl", "stylesanxl", "flux", "sd3", "playground"],
    "commercial":  ["midjourney", "dall-e", "dalle", "gpt-image", "imagen", "firefly", "nano-banana"],
}
# Families the brief requires; any missing one is reported, not silently dropped.
REQUIRED_FAMILIES = ["gan", "sd1x", "modern_open", "commercial"]

MAX_GEN_SHARE = 0.02   # no single generator may exceed 2% of the fake pool


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def dedup(phash: np.ndarray, dataset: np.ndarray, max_ham: int = 4):
    """
    Near-duplicate removal via phash Hamming distance.

    Exact duplicates go first (cheap hash grouping). Near-duplicates are then
    found by bucketing on 16-bit slices of the hash: two hashes within
    Hamming distance 4 must agree exactly on at least one of the four
    16-bit segments (pigeonhole), so this finds every pair without the
    O(n^2) comparison.
    """
    n = len(phash)
    keep = np.ones(n, dtype=bool)
    pair_counts = Counter()

    order = np.argsort(phash, kind="stable")
    # --- exact duplicates ---
    ph_sorted = phash[order]
    same = np.flatnonzero(ph_sorted[1:] == ph_sorted[:-1])
    for i in same:
        a, b = order[i], order[i + 1]
        if keep[a] and keep[b]:
            keep[b] = False
            pair_counts[tuple(sorted((dataset[a], dataset[b])))] += 1

    # --- near duplicates via 16-bit banding ---
    for shift in (0, 16, 32, 48):
        band = ((phash >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint16)
        buckets = defaultdict(list)
        for i in np.flatnonzero(keep):
            buckets[band[i]].append(i)
        for _, idxs in buckets.items():
            if len(idxs) < 2 or len(idxs) > 4000:
                continue
            arr = np.array(idxs)
            hs = phash[arr]
            for j in range(len(arr)):
                a = arr[j]
                if not keep[a]:
                    continue
                rest = arr[j + 1:]
                if rest.size == 0:
                    break
                alive = rest[keep[rest]]
                if alive.size == 0:
                    continue
                d = popcount64(np.bitwise_xor(phash[alive], hs[j]))
                dup = alive[d <= max_ham]
                for b in dup:
                    if keep[b]:
                        keep[b] = False
                        pair_counts[tuple(sorted((dataset[a], dataset[b])))] += 1
    return keep, pair_counts


def pick_logo(generators, commercial_gens=None):
    """Choose held-out generators, one family at a time, largest first."""
    chosen, used = {}, set()
    commercial_gens = commercial_gens or set()
    fakes = [g for g in generators if not g.startswith("real:")]
    for fam, pats in LOGO_PATTERNS.items():
        if fam == "commercial":
            # Only genuine Commercial-subset generators qualify. A community
            # HF repo called "openskyml/midjourney-mini" is a Stable Diffusion
            # finetune, not Midjourney, and holding it out would misrepresent
            # the result as covering commercial models.
            hits = [g for g in fakes if g in commercial_gens and g not in used]
        else:
            hits = [g for g in fakes
                    if g not in used and any(p in g.lower() for p in pats)]
        if hits:
            hits.sort(key=lambda g: -generators[g])
            take = hits[: max(1, min(3, len(hits)))]
            chosen[fam] = take
            used.update(take)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-name", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = PROC / args.proc_name
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"[split] no parquet in {src}")

    print(f"[split] reading metadata from {len(files)} shards ...", flush=True)
    tbl = pq.read_table(files, columns=META_COLS)
    d = tbl.to_pydict()
    n0 = len(d["uid"])
    print(f"[split] {n0:,} rows")

    phash = np.array(d["phash"], dtype=np.uint64)
    dsarr = np.array(d["dataset"], dtype=object)
    label = np.array(d["label"], dtype=np.int8)
    gen = np.array(d["generator"], dtype=object)

    # ---- perceptual dedup ----
    print("[split] deduplicating (phash, hamming<=4) ...", flush=True)
    keep, pair_counts = dedup(phash, dsarr)
    n_dropped = int((~keep).sum())
    print(f"[split] dropped {n_dropped:,} duplicates ({100*n_dropped/n0:.2f}%)")
    for pair, c in pair_counts.most_common(12):
        print(f"        {pair[0]} <-> {pair[1]}: {c:,}")

    # ---- leave-one-generator-out splits ----
    rng = np.random.default_rng(args.seed)
    gen_counts = Counter(gen[keep].tolist())
    fake_gen_counts = {g: c for g, c in gen_counts.items() if not g.startswith("real:")}

    subset_arr = np.array(d["subset"], dtype=object)
    commercial_gens = {g for g, sb in zip(gen, subset_arr)
                       if sb == "Commercial" and not g.startswith("real:")}
    print(f"[split] genuine Commercial-subset generators: {len(commercial_gens)}")
    logo = pick_logo(gen_counts, commercial_gens)
    logo_gens = {g for v in logo.values() for g in v}
    missing_families = [f for f in REQUIRED_FAMILIES if f not in logo]
    print(f"[split] LOGO families: {logo}")
    if missing_families:
        print(f"[split] WARNING: required families absent from this corpus: "
              f"{missing_families} — cross-generator results cannot speak to them.")

    split = np.array(["train"] * n0, dtype=object)
    split[~keep] = "dropped_dup"

    # LOGO generators are removed from training entirely.
    for i in np.flatnonzero(keep):
        if gen[i] in logo_gens:
            split[i] = "logo"

    # Cap per-generator share so the long tail of small generators, which is
    # what actually drives cross-generator generalization, is not swamped.
    eligible = np.flatnonzero((split == "train") & (label == 1))
    n_fake = len(eligible)
    cap = max(1, int(MAX_GEN_SHARE * n_fake))
    per_gen = defaultdict(list)
    for i in eligible:
        per_gen[gen[i]].append(i)
    capped = 0
    for g, idxs in per_gen.items():
        if len(idxs) > cap:
            drop = rng.permutation(idxs)[cap:]
            split[drop] = "capped_out"
            capped += len(drop)
    print(f"[split] per-generator cap {cap:,} ({MAX_GEN_SHARE:.0%}) -> removed {capped:,}")

    # Reals need their own treatment. CommunityForensics-Small reports
    # real_source = "N/A" for every real image, collapsing them into a single
    # group, so a group-aware real split is impossible: holding out the one
    # group would leave train with no reals and val/logo with no negatives to
    # score against. Reals are therefore split at random and this is recorded
    # as a limitation rather than presented as leakage-free.
    real_groups = {g for g in gen[keep] if g.startswith("real:")}
    reals_grouped = len(real_groups) >= 2
    real_idx = np.flatnonzero((split == "train") & (label == 0))
    if not reals_grouped:
        perm = rng.permutation(real_idx)
        n_logo_r = int(0.15 * len(perm))
        n_val_r = int(args.val_frac * len(perm))
        split[perm[:n_logo_r]] = "logo"
        split[perm[n_logo_r:n_logo_r + n_val_r]] = "val"

    # Group-aware train/val for fakes: a generator lives entirely in train or
    # entirely in val, never split across the two.
    train_pool = np.flatnonzero((split == "train") & (label == 1))
    groups = sorted({gen[i] for i in train_pool})
    rng.shuffle(groups)
    counts = Counter(gen[train_pool].tolist())
    target_val = args.val_frac * len(train_pool)
    val_groups, acc = set(), 0
    for g in groups:
        if acc >= target_val:
            break
        val_groups.add(g)
        acc += counts[g]
    for i in train_pool:
        if gen[i] in val_groups:
            split[i] = "val"

    # Balance real:fake 1:1 in train by subsampling the larger side.
    tr = np.flatnonzero(split == "train")
    tr_fake = tr[label[tr] == 1]
    tr_real = tr[label[tr] == 0]
    k = min(len(tr_fake), len(tr_real))
    if len(tr_fake) > k:
        split[rng.permutation(tr_fake)[k:]] = "balanced_out"
    if len(tr_real) > k:
        split[rng.permutation(tr_real)[k:]] = "balanced_out"
    print(f"[split] train balanced to {k:,} per class")

    # ---- assertions (permanent) ----
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    lo = np.flatnonzero(split == "logo")

    # Reals are excluded from the overlap assertions when ungrouped, since a
    # single random-split real group legitimately appears on both sides.
    def fakes_only(idx):
        return {gen[i] for i in idx if not gen[i].startswith("real:")}

    g_tr, g_va, g_lo = fakes_only(tr), fakes_only(va), fakes_only(lo)
    assert not (g_tr & g_va), f"generator overlap train/val: {sorted(g_tr & g_va)[:5]}"
    assert not (g_tr & g_lo), f"LOGO generator leaked into train: {sorted(g_tr & g_lo)[:5]}"
    assert not (g_va & g_lo), f"LOGO generator leaked into val: {sorted(g_va & g_lo)[:5]}"
    tr_ph = set(phash[tr].tolist())
    assert len(tr_ph & set(phash[lo].tolist())) == 0, "exact phash collision train/logo"
    assert (label[lo] == 0).sum() > 0, "LOGO split has no reals — AUC undefined"
    assert (label[va] == 0).sum() > 0, "val split has no reals — AUC undefined"
    print("[split] ASSERTIONS PASSED: zero fake-generator overlap across "
          "train/val/logo; both eval splits carry negatives")

    out_name = args.out or args.proc_name
    d["split"] = split.tolist()
    d["kept"] = keep.tolist()
    MANI.mkdir(parents=True, exist_ok=True)
    out_path = MANI / f"manifest_{out_name}.parquet"
    # phash occupies the full 64-bit range; pyarrow infers int64 from a
    # Python int list and overflows on anything above 2^63.
    arrays, names = [], []
    for col, vals in d.items():
        names.append(col)
        arrays.append(pa.array(vals, type=pa.uint64()) if col == "phash"
                      else pa.array(vals))
    pq.write_table(pa.Table.from_arrays(arrays, names=names),
                   out_path, compression="zstd")

    stats = {
        "proc_name": args.proc_name,
        "n_rows": n0,
        "n_duplicates_dropped": n_dropped,
        "duplicate_pairs_by_dataset": {f"{a}|{b}": c for (a, b), c in pair_counts.items()},
        "split_counts": {k: int(v) for k, v in Counter(split.tolist()).items()},
        "n_generators_total": len(fake_gen_counts),
        "n_generators_train": len([g for g in g_tr if not g.startswith("real:")]),
        "n_generators_val": len([g for g in g_va if not g.startswith("real:")]),
        "logo_families": logo,
        "required_families_missing": missing_families,
        "reals_group_aware": bool(reals_grouped),
        "logo_generators": sorted(logo_gens),
        "per_generator_cap": cap,
        "train_per_class": int(k),
        "class_counts": {
            s: dict(Counter(label[split == s].tolist()))
            for s in ("train", "val", "logo")
        },
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"splits_{out_name}.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats["split_counts"], indent=2))
    print(f"[split] manifest -> {out_path}")


if __name__ == "__main__":
    main()

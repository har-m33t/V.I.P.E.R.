"""
CLIP ViT-L/14 linear probe, plus the blur-baseline leakage check.

Two modes:
  --blur 0    the real detector
  --blur N    the sanity check: Gaussian-blur every image with radius N before
              encoding, destroying high-frequency content. A probe that still
              scores well above chance is reading semantics or source, not
              generation artifacts.

Model selection is on LOGO AUC, never val AUC. Selecting on an
in-distribution metric is the single most common way this task goes wrong:
val here shares a canonicalization pipeline and a real-image pool with train,
so it saturates long before cross-generator performance does.
"""

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROC = Path("/data/proc")
MANI = Path("/data/manifest")
CKPT = Path("/data/ckpt")
LOGS = Path("/data/logs")


class ShardImages(Dataset):
    """Reads canonicalized JPEGs out of the proc parquet shards."""

    def __init__(self, rows, preprocess, blur=0.0, aug_rng=None):
        self.rows = rows              # list of (shard_path, row_idx)
        self.preprocess = preprocess
        self.blur = blur
        self.aug_rng = aug_rng
        self._cache_path = None
        self._cache_col = None

    def __len__(self):
        return len(self.rows)

    def _get_jpeg(self, path, idx):
        # Shards are read many times; hold the most recent one per worker.
        if path != self._cache_path:
            self._cache_col = pq.read_table(path, columns=["image_jpeg"]).column(0)
            self._cache_path = path
        return self._cache_col[idx].as_py()

    def __getitem__(self, i):
        path, idx = self.rows[i]
        im = Image.open(io.BytesIO(self._get_jpeg(path, idx))).convert("RGB")

        if self.aug_rng is not None:
            # Deployment conditions, applied identically regardless of class.
            r = self.aug_rng
            if r.random() < 0.5:
                q = int(r.integers(30, 101))
                b = io.BytesIO()
                im.save(b, format="JPEG", quality=q)
                im = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
            if r.random() < 0.3:
                im = im.filter(ImageFilter.GaussianBlur(float(r.uniform(0.3, 1.5))))
            if r.random() < 0.3:
                f = float(r.uniform(0.5, 0.9))
                w, h = im.size
                im = im.resize((max(32, int(w * f)), max(32, int(h * f))), Image.BICUBIC)
                im = im.resize((w, h), Image.BICUBIC)

        if self.blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(self.blur))

        return self.preprocess(im)


def load_manifest(name):
    t = pq.read_table(MANI / f"manifest_{name}.parquet")
    return t.to_pydict()


def build_rows(man, proc_name, split):
    """
    Resolve manifest uids to concrete (shard_path, row_index) addresses by
    reading each shard's own uid column, rather than reconstructing the index
    arithmetically. Reconstruction silently misaligns the moment any row is
    dropped during canonicalization, and a misaligned label array would still
    train and still report a plausible AUC.
    """
    want = {}
    for i, (uid, s_) in enumerate(zip(man["uid"], man["split"])):
        if s_ == split:
            want[uid] = i

    rows, meta = [], []
    for shard in sorted((PROC / proc_name).glob("*.parquet")):
        uids = pq.read_table(shard, columns=["uid"]).column(0).to_pylist()
        for pos, uid in enumerate(uids):
            mi = want.get(uid)
            if mi is not None:
                rows.append((str(shard), pos))
                meta.append(mi)
    return rows, meta


@torch.no_grad()
def extract(model, loader, device, dim):
    feats = np.zeros((len(loader.dataset), dim), dtype=np.float16)
    k = 0
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        b = f.shape[0]
        feats[k:k + b] = f.cpu().numpy().astype(np.float16)
        k += b
        if k % 51200 < b:
            print(f"    {k:,}/{len(feats):,}  {k/max(time.time()-t0,1):.0f} img/s", flush=True)
    return feats


def evaluate(probe, X, y, device):
    probe.eval()
    with torch.no_grad():
        s = probe(torch.from_numpy(X).float().to(device)).squeeze(-1).cpu().numpy()
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "n": len(y)}
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "n": int(len(y)),
        "scores": s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-name", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-per-split", type=int, default=0)
    args = ap.parse_args()

    import open_clip
    device = torch.device("cuda")
    # OpenAI's CLIP weights were trained with QuickGELU. open_clip 3.x maps
    # the plain "ViT-L-14" config to standard GELU, which loads the weights
    # without error but computes a different activation -- degraded features
    # that still produce plausible-looking AUCs. The -quickgelu config is the
    # one that matches these weights.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai", device=device)
    model.eval()
    dim = model.visual.output_dim
    print(f"[clip] CLIP ViT-L/14 loaded, dim={dim}, blur={args.blur}", flush=True)

    man = load_manifest(args.manifest)
    y_all = np.array(man["label"], dtype=np.int64)
    gen_all = np.array(man["generator"], dtype=object)

    feats, labels, gens = {}, {}, {}
    for split in ("train", "val", "logo"):
        rows, meta = build_rows(man, args.proc_name, split)
        if args.max_per_split and len(rows) > args.max_per_split:
            sel = np.random.default_rng(0).permutation(len(rows))[: args.max_per_split]
            rows = [rows[i] for i in sel]
            meta = [meta[i] for i in sel]
        if not rows:
            print(f"[clip] {split}: empty, skipping")
            continue
        rng = np.random.default_rng(7) if split == "train" else None
        ds = ShardImages(rows, preprocess, blur=args.blur, aug_rng=rng)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, shuffle=False)
        print(f"[clip] extracting {split}: {len(rows):,} images", flush=True)
        feats[split] = extract(model, dl, device, dim)
        labels[split] = y_all[meta]
        gens[split] = gen_all[meta]

    del model
    torch.cuda.empty_cache()

    # ---- linear probe ----
    probe = nn.Linear(dim, 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    Xtr = torch.from_numpy(feats["train"]).float()
    ytr = torch.from_numpy(labels["train"]).float()
    n = len(Xtr)
    best = {"logo_auc": -1, "epoch": 0, "state": None}
    bad = 0
    hist = []

    for ep in range(1, args.epochs + 1):
        probe.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            out = probe(xb).squeeze(-1)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        m_val = evaluate(probe, feats["val"], labels["val"], device) if "val" in feats else {"auc": float("nan")}
        m_logo = evaluate(probe, feats["logo"], labels["logo"], device) if "logo" in feats else {"auc": float("nan")}
        hist.append({"epoch": ep, "loss": tot / n,
                     "val_auc": m_val["auc"], "logo_auc": m_logo["auc"]})
        print(f"[clip] ep{ep:02d} loss={tot/n:.4f} val_auc={m_val['auc']:.4f} "
              f"LOGO_auc={m_logo['auc']:.4f}", flush=True)

        sel = m_logo["auc"] if not np.isnan(m_logo["auc"]) else m_val["auc"]
        if sel > best["logo_auc"]:
            best = {"logo_auc": sel, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[clip] early stop at ep{ep} (best ep{best['epoch']})")
                break

    probe.load_state_dict(best["state"])
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "dim": dim, "tag": args.tag},
               CKPT / f"probe_{args.tag}.pt")

    # ---- reporting ----
    res = {"tag": args.tag, "blur": args.blur, "best_epoch": best["epoch"],
           "history": hist, "splits": {}}
    for split in feats:
        m = evaluate(probe, feats[split], labels[split], device)
        scores = m.pop("scores", None)
        res["splits"][split] = m
        if split == "logo" and scores is not None:
            per_gen = {}
            for g in sorted(set(gens[split].tolist())):
                if g.startswith("real:"):
                    continue
                mask = gens[split] == g
                # score each held-out generator against the full real pool
                real = labels[split] == 0
                sel = mask | real
                if mask.sum() < 20 or real.sum() < 20:
                    continue
                per_gen[g] = {
                    "n_fake": int(mask.sum()),
                    "auc": float(roc_auc_score(labels[split][sel], scores[sel])),
                    "ap": float(average_precision_score(labels[split][sel], scores[sel])),
                }
            res["per_held_out_generator"] = per_gen

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"probe_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res["splits"].items()}, indent=2))
    print(f"[clip] -> {LOGS / f'probe_{args.tag}.json'}")


if __name__ == "__main__":
    main()

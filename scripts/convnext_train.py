"""
ConvNeXt-Tiny (the original VIPER backbone), image-only.

Mirrors src/model.py's ConvNeXtBaseline exactly: torchvision ConvNeXt-Tiny with
IMAGENET1K_V1 weights, classifier replaced by Dropout(0.3) + Linear(768, 2),
backbone frozen except the last two stage groups, Adam at 1e-4 with cosine
annealing, CrossEntropyLoss, batch 64. Reimplemented inline rather than
imported so the pod needs no src/ tree, but the layer-for-layer construction
and the unfreeze boundary are the same.

Differs from the 67k run in exactly two ways, both required to make the
comparison against the CLIP probe meaningful:
  - trained and evaluated on the same leakage-free splits (identical manifest)
  - selected on LOGO AUC rather than val F1, matching the probe's protocol
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

PROC = Path("/data/proc")
MANI = Path("/data/manifest")
CKPT = Path("/data/ckpt")
LOGS = Path("/data/logs")
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ConvNeXtBaseline(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, unfreeze_stages=2, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        self.embedding_norm = self.backbone.classifier[0]
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(768, num_classes))

        self._stage_groups = [
            (self.backbone.features[0], self.backbone.features[1]),
            (self.backbone.features[2], self.backbone.features[3]),
            (self.backbone.features[4], self.backbone.features[5]),
            (self.backbone.features[6], self.backbone.features[7]),
        ]
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.embedding_norm.parameters():
            p.requires_grad = True
        for grp in self._stage_groups[-unfreeze_stages:]:
            for mod in grp:
                for p in mod.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, x):
        f = self.backbone.features(x)
        pooled = self.backbone.avgpool(f)
        return self.head(torch.flatten(self.embedding_norm(pooled), 1))


def preload_jpegs(rows):
    """
    Pull every needed JPEG into RAM up front.

    The obvious lazy alternative -- cache the most recent shard per worker --
    collapses under shuffle=True: consecutive samples land in different
    shards, so every __getitem__ re-reads a whole parquet column. That capped
    the first attempt at 74 img/s. At ~26 KB per canonicalized image, 80k
    images is ~2 GB, which this box has in abundance.
    """
    by_shard = {}
    for i, (path, idx) in enumerate(rows):
        by_shard.setdefault(path, []).append((idx, i))
    out = [None] * len(rows)
    for path, items in by_shard.items():
        col = pq.read_table(path, columns=["image_jpeg"]).column(0)
        for idx, i in items:
            out[i] = col[idx].as_py()
    return out


class ShardImages(Dataset):
    def __init__(self, rows, labels, train=False):
        self.blobs, self.labels = preload_jpegs(rows), labels
        self.t = (transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
            if train else transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        im = Image.open(io.BytesIO(self.blobs[i])).convert("RGB")
        return self.t(im), int(self.labels[i])


def build_rows(man, proc_name, split, cap, seed=0):
    want = {u: i for i, (u, s) in enumerate(zip(man["uid"], man["split"])) if s == split}
    rows, meta = [], []
    for shard in sorted((PROC / proc_name).glob("*.parquet")):
        for pos, uid in enumerate(pq.read_table(shard, columns=["uid"]).column(0).to_pylist()):
            mi = want.get(uid)
            if mi is not None:
                rows.append((str(shard), pos))
                meta.append(mi)
    if cap and len(rows) > cap:
        sel = np.random.default_rng(seed).permutation(len(rows))[:cap]
        rows = [rows[i] for i in sel]
        meta = [meta[i] for i in sel]
    return rows, np.array(meta)


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        ps.append(torch.softmax(out.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-name", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-cap", type=int, default=80000)
    ap.add_argument("--eval-cap", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    man = pq.read_table(MANI / f"manifest_{args.manifest}.parquet").to_pydict()
    y_all = np.array(man["label"], dtype=np.int64)
    gen_all = np.array(man["generator"], dtype=object)

    loaders, metas = {}, {}
    for split, cap, tr in (("train", args.train_cap, True),
                           ("val", args.eval_cap, False),
                           ("logo", args.eval_cap, False)):
        rows, meta = build_rows(man, args.proc_name, split, cap)
        metas[split] = meta
        loaders[split] = DataLoader(
            ShardImages(rows, y_all[meta], train=tr), batch_size=args.batch,
            shuffle=tr, num_workers=args.workers, pin_memory=True, drop_last=tr,
            persistent_workers=True, prefetch_factor=4)
        print(f"[convnext] {split}: {len(rows):,}", flush=True)

    model = ConvNeXtBaseline().to(device).to(memory_format=torch.channels_last)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[convnext] ConvNeXt-Tiny trainable {trainable:,}/{total:,}", flush=True)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best = {"logo_auc": -1, "epoch": 0, "state": None}
    bad, hist = 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()

        pv, yv = infer(model, loaders["val"], device)
        pl, yl = infer(model, loaders["logo"], device)
        va = roc_auc_score(yv, pv)
        la = roc_auc_score(yl, pl)
        el = time.time() - t0
        hist.append({"epoch": ep, "loss": tot / seen, "val_auc": float(va),
                     "logo_auc": float(la), "seconds": el,
                     "img_per_s": seen / el})
        print(f"[convnext] ep{ep:02d} loss={tot/seen:.4f} val_auc={va:.4f} "
              f"LOGO_auc={la:.4f} [{el/60:.1f}min {seen/el:.0f} img/s]", flush=True)

        if la > best["logo_auc"]:
            best = {"logo_auc": float(la), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[convnext] early stop ep{ep} (best ep{best['epoch']})", flush=True)
                break

    model.load_state_dict(best["state"])
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "tag": args.tag}, CKPT / f"convnext_{args.tag}.pt")

    res = {"tag": args.tag, "arch": "convnext_tiny", "best_epoch": best["epoch"],
           "trainable_params": trainable, "total_params": total, "history": hist, "splits": {}}
    for split in ("val", "logo"):
        p, y = infer(model, loaders[split], device)
        res["splits"][split] = {"auc": float(roc_auc_score(y, p)),
                                "ap": float(average_precision_score(y, p)), "n": int(len(y))}
        if split == "logo":
            g = gen_all[metas["logo"]]
            per = {}
            for gg in sorted(set(g.tolist())):
                if gg.startswith("real:"):
                    continue
                m = g == gg
                r = y == 0
                if m.sum() < 20 or r.sum() < 20:
                    continue
                sel = m | r
                per[gg] = {"n_fake": int(m.sum()),
                           "auc": float(roc_auc_score(y[sel], p[sel])),
                           "ap": float(average_precision_score(y[sel], p[sel]))}
            res["per_held_out_generator"] = per

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"convnext_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["splits"], indent=2))


if __name__ == "__main__":
    main()

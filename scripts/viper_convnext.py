"""
VIPER-ConvNeXt: ConvNeXt-Tiny backbone + forensic features + MLP head.

Same backbone and schedule as convnext_train.py, with src/model.py's forensic fusion
head switched on: the 768-d ConvNeXt embedding is concatenated with the
standardized forensic vector and routed through Linear(768+d, 384) -> ReLU ->
Linear(384, 2). Omitting --features reproduces the image-only arm exactly,
so every arm in the comparison runs from one script.

Two orthogonal switches define the grid:
  --unfreeze-stages  2 = fine-tune the last two ConvNeXt stage groups (default),
                     0 = fully frozen feature extractor, head-only training
  --mlp-head         force the 2-layer head with no forensic features, so the
                     fusion delta is not confounded with head capacity

Fusion is expected to help most at --unfreeze-stages 0: a frozen ImageNet
backbone cannot learn the low-level noise and gradient statistics the forensic
vector supplies, whereas a fine-tuned one derives them from the same pixels.

Forensic features are standardized with statistics fitted on the TRAIN split
only. Fitting the scaler over the whole corpus leaks val/LOGO distribution
into training and would quietly inflate the very numbers this run exists to
measure.

Mirrors src/model.py's VIPERConvNeXt exactly: torchvision ConvNeXt-Tiny with
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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
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


class VIPERConvNeXt(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, unfreeze_stages=2,
                 pretrained=True, eda_feature_dim=0, fusion_hidden_dim=384,
                 mlp_head=False):
        super().__init__()
        self.eda_feature_dim = int(eda_feature_dim)
        self.frozen = int(unfreeze_stages) == 0
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        self.embedding_norm = self.backbone.classifier[0]
        self.backbone.classifier = nn.Identity()
        # mlp_head decouples head capacity from the presence of forensic
        # features, so a fusion win cannot be an MLP-vs-linear win in disguise.
        if self.eda_feature_dim > 0 or mlp_head:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(768 + self.eda_feature_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_hidden_dim, num_classes))
        else:
            self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(768, num_classes))

        self._stage_groups = [
            (self.backbone.features[0], self.backbone.features[1]),
            (self.backbone.features[2], self.backbone.features[3]),
            (self.backbone.features[4], self.backbone.features[5]),
            (self.backbone.features[6], self.backbone.features[7]),
        ]
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        # A fully frozen backbone must also freeze the final LayerNorm, or the
        # "frozen" arm still adapts 1,536 params of the representation.
        for p in self.embedding_norm.parameters():
            p.requires_grad = not self.frozen
        # lst[-0:] is the whole list, not the empty slice -- guard it.
        for grp in (self._stage_groups[-unfreeze_stages:] if unfreeze_stages else []):
            for mod in grp:
                for p in mod.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        """Keep a frozen backbone in eval mode.

        ConvNeXt's stochastic-depth layers stay active in train mode. On a
        frozen extractor that injects noise into features nothing can adapt
        to -- the head would be fitting a moving target for no reason.
        """
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
            self.embedding_norm.eval()
        return self

    def _embed(self, x):
        f = self.backbone.features(x)
        pooled = self.backbone.avgpool(f)
        return torch.flatten(self.embedding_norm(pooled), 1)

    def forward(self, x, eda=None):
        if self.frozen:
            with torch.no_grad():
                emb = self._embed(x)
            emb = emb.detach()
        else:
            emb = self._embed(x)
        if self.eda_feature_dim > 0:
            if eda is None:
                raise ValueError("fusion model called without forensic features")
            emb = torch.cat([emb, eda.to(emb.dtype)], dim=1)
        return self.head(emb)


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
    def __init__(self, rows, labels, train=False, feats=None):
        self.blobs, self.labels = preload_jpegs(rows), labels
        self.feats = feats
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
        x = self.t(im)
        if self.feats is None:
            return x, int(self.labels[i])
        return x, torch.from_numpy(self.feats[i]), int(self.labels[i])


def load_features(feat_dir):
    """uid -> forensic vector, plus the ordered feature names."""
    files = sorted(Path(feat_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"[viper] no feature parquet in {feat_dir}")
    names = [c for c in pq.ParquetFile(files[0]).schema_arrow.names
             if c.startswith("f_")]
    fmap = {}
    for f in files:
        t = pq.read_table(f, columns=["uid"] + names)
        uids = t.column("uid").to_pylist()
        mat = np.column_stack([t.column(c).to_numpy(zero_copy_only=False)
                               for c in names]).astype(np.float32)
        for u, v in zip(uids, mat):
            fmap[u] = v
    print(f"[viper] loaded forensic features: {len(fmap):,} uids x {len(names)} dims")
    return fmap, names


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


def _unpack(batch, device):
    if len(batch) == 3:
        x, e, y = batch
        e = e.to(device, non_blocking=True).float()
    else:
        x, y = batch
        e = None
    x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return x, e, y


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ps, ys = [], []
    for batch in loader:
        x, e, y = _unpack(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, e)
        ps.append(torch.softmax(out.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def threshold_metrics(y, p):
    """Ranking metrics plus hard-label metrics at two thresholds.

    AUC/AP alone hide the operating point. 0.5 is what a deployed model
    actually uses; the best-F1 threshold shows the ceiling that calibration
    could reach, so a gap between them is a calibration problem rather than a
    discrimination one.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    out = {"n": int(len(y)), "n_ai": int((y == 1).sum()), "n_real": int((y == 0).sum()),
           "auc": float(roc_auc_score(y, p)),
           "ap": float(average_precision_score(y, p)),
           "brier": float(np.mean((p - y) ** 2)),
           "majority_baseline_accuracy": float(max((y == 1).mean(), (y == 0).mean()))}

    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve returns one more point than thresholds.
    best_i = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
    for name, t in (("at_0.5", 0.5), ("at_best_f1", float(thr[best_i]) if len(thr) else 0.5)):
        yh = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        out[name] = {
            "threshold": float(t),
            "accuracy": float(accuracy_score(y, yh)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yh)),
            "precision_ai": float(precision_score(y, yh, pos_label=1, zero_division=0)),
            "recall_ai": float(recall_score(y, yh, pos_label=1, zero_division=0)),
            "f1_ai": float(f1_score(y, yh, pos_label=1, zero_division=0)),
            "precision_real": float(precision_score(y, yh, pos_label=0, zero_division=0)),
            "recall_real": float(recall_score(y, yh, pos_label=0, zero_division=0)),
            "f1_real": float(f1_score(y, yh, pos_label=0, zero_division=0)),
            "f1_macro": float(f1_score(y, yh, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y, yh)),
            "fpr": float(fp / max(fp + tn, 1)),
            "fnr": float(fn / max(fn + tp, 1)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    return out


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
    ap.add_argument("--features", default=None,
                    help="dir of per-shard forensic parquet; omit for image-only")
    ap.add_argument("--unfreeze-stages", type=int, default=2,
                    help="trailing ConvNeXt stage groups to fine-tune; 0 = frozen backbone")
    ap.add_argument("--mlp-head", action="store_true",
                    help="use the 2-layer head even with no forensic features")
    ap.add_argument("--score-only", default=None,
                    help="path to a checkpoint: skip training, just evaluate it")
    ap.add_argument("--dump-uids", default=None,
                    help="write the exact uids this config would train/eval on, then exit")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    man = pq.read_table(MANI / f"manifest_{args.manifest}.parquet").to_pydict()
    y_all = np.array(man["label"], dtype=np.int64)
    gen_all = np.array(man["generator"], dtype=object)

    if args.dump_uids:
        # Emit the precise uid set this configuration selects, so forensic
        # features are computed for exactly these rows and the fusion arm
        # cannot silently train on a different sample than the image-only arm.
        uid_all_d = np.array(man["uid"], dtype=object)
        out = []
        for split, cap in (("train", args.train_cap), ("val", args.eval_cap),
                           ("logo", args.eval_cap)):
            _, meta = build_rows(man, args.proc_name, split, cap)
            out.extend(uid_all_d[meta].tolist())
        Path(args.dump_uids).write_text("\n".join(out))
        print(f"[viper] wrote {len(out):,} uids -> {args.dump_uids}")
        return

    fmap, fnames = (load_features(args.features) if args.features else (None, []))
    feat_dim = len(fnames)
    uid_all = np.array(man["uid"], dtype=object)

    raw_feats, metas, rowsets = {}, {}, {}
    for split, cap in (("train", args.train_cap), ("val", args.eval_cap),
                       ("logo", args.eval_cap)):
        rows, meta = build_rows(man, args.proc_name, split, cap)
        rowsets[split], metas[split] = rows, meta
        if fmap is not None:
            missing = [u for u in uid_all[meta] if u not in fmap]
            if missing:
                raise SystemExit(f"[viper] {len(missing):,} {split} uids lack forensic "
                                 f"features, e.g. {missing[:3]}")
            raw_feats[split] = np.stack([fmap[u] for u in uid_all[meta]]).astype(np.float32)
        print(f"[viper] {split}: {len(rows):,}", flush=True)

    if fmap is not None:
        # Scaler fitted on train only -- see module docstring.
        mu = raw_feats["train"].mean(0)
        sd = raw_feats["train"].std(0)
        sd[sd < 1e-8] = 1.0
        for k in raw_feats:
            raw_feats[k] = np.clip((raw_feats[k] - mu) / sd, -10.0, 10.0).astype(np.float32)
        print(f"[viper] forensic features: {feat_dim} dims, standardized on train "
              f"(n={len(raw_feats['train']):,})", flush=True)

    loaders = {}
    for split, tr in (("train", True), ("val", False), ("logo", False)):
        loaders[split] = DataLoader(
            ShardImages(rowsets[split], y_all[metas[split]], train=tr,
                        feats=raw_feats.get(split)),
            batch_size=args.batch, shuffle=tr, num_workers=args.workers,
            pin_memory=True, drop_last=tr, persistent_workers=True, prefetch_factor=4)

    model = (VIPERConvNeXt(eda_feature_dim=feat_dim,
                             unfreeze_stages=args.unfreeze_stages,
                             mlp_head=args.mlp_head)
             .to(device).to(memory_format=torch.channels_last))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    head_kind = "mlp" if (feat_dim > 0 or args.mlp_head) else "linear"
    # "VIPER" names the full architecture -- backbone + forensic features + MLP
    # head. An arm missing the forensic branch is an ablation control, not a
    # VIPER model, so it is named for what it actually is.
    state = "Frozen" if args.unfreeze_stages == 0 else "Unfrozen"
    head_name = "MLP" if head_kind == "mlp" else "Linear"
    if feat_dim > 0:
        model_name = f"VIPER-ConvNeXt-{state}-{head_name}"
    else:
        model_name = f"ConvNeXt-{state}-{head_name} (no forensic)"
    print(f"[viper] {model_name}: trainable {trainable:,}/{total:,} "
          f"unfreeze_stages={args.unfreeze_stages} head={head_kind} "
          f"feat_dim={feat_dim} lr={args.lr}", flush=True)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best = {"logo_auc": -1, "epoch": 0, "state": None}
    bad, hist = 0, []
    if args.score_only:
        sd = torch.load(args.score_only, map_location="cpu")["state_dict"]
        model.load_state_dict(sd)
        best = {"logo_auc": -1, "epoch": -1, "state": sd}
        print(f"[viper] scoring {args.score_only} -- no training", flush=True)
    for ep in ([] if args.score_only else range(1, args.epochs + 1)):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for batch in loaders["train"]:
            x, e, y = _unpack(batch, device)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x, e), y)
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
        print(f"[viper] ep{ep:02d} loss={tot/seen:.4f} val_auc={va:.4f} "
              f"LOGO_auc={la:.4f} [{el/60:.1f}min {seen/el:.0f} img/s]", flush=True)

        if la > best["logo_auc"]:
            best = {"logo_auc": float(la), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[viper] early stop ep{ep} (best ep{best['epoch']})", flush=True)
                break

    model.load_state_dict(best["state"])
    if not args.score_only:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best["state"], "tag": args.tag}, CKPT / f"{args.tag}.pt")

    res = {"tag": args.tag, "model_name": model_name, "arch": "convnext_tiny",
           "fusion": feat_dim > 0, "feature_dim": feat_dim,
           "unfreeze_stages": args.unfreeze_stages,
           "backbone_frozen": args.unfreeze_stages == 0, "head": head_kind,
           "lr": args.lr, "epochs": args.epochs,
           "feature_names": fnames, "best_epoch": best["epoch"],
           "trainable_params": trainable, "total_params": total, "history": hist, "splits": {}}
    scores = {}
    for split in ("val", "logo"):
        p, y = infer(model, loaders[split], device)
        scores[f"{split}_p"], scores[f"{split}_y"] = p, y
        res["splits"][split] = threshold_metrics(y, p)
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
    (LOGS / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    # Persist raw per-sample scores: any threshold, calibration curve or ROC
    # overlay wanted later is then a local computation, not another pod.
    np.savez_compressed(LOGS / f"{args.tag}_scores.npz", **scores)
    print(json.dumps(res["splits"], indent=2))


if __name__ == "__main__":
    main()

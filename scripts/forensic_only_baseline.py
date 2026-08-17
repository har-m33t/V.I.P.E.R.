"""Forensic features alone: how much signal do the 33 hand-crafted features
carry with no CNN at all?

This is the control that separates "redundant" from "useless". If the fusion
head adds nothing on top of a fine-tuned backbone, that could mean the features
are noise -- or that the backbone already derives them. A logistic regression on
the features by themselves tells you which.

Scores use predict_proba, not decision_function: threshold metrics need
probabilities, and the two agree on AUC anyway.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from viper_convnext import threshold_metrics

MANI = Path("/data/manifest")
FEAT = Path("/data/features")
LOGS = Path("/data/logs")


def main():
    man = pq.read_table(MANI / "manifest_cfsmall_v1.parquet").to_pydict()
    uid2split = dict(zip(man["uid"], man["split"]))
    uid2label = dict(zip(man["uid"], man["label"]))

    files = sorted((FEAT / "cfsmall_v1").glob("*.parquet"))
    names = [c for c in pq.ParquetFile(files[0]).schema_arrow.names if c.startswith("f_")]
    X, y, sp = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["uid"] + names)
        uids = t.column("uid").to_pylist()
        m = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in names])
        for u, v in zip(uids, m):
            s = uid2split.get(u)
            if s in ("train", "val", "logo"):
                X.append(v)
                y.append(uid2label[u])
                sp.append(s)
    X = np.asarray(X, np.float64)
    y = np.asarray(y)
    sp = np.asarray(sp)
    print(f"[forensic-only] loaded {len(X):,} x {X.shape[1]}", flush=True)

    # Scaler fitted on train only, matching the fusion arms exactly.
    tr = sp == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = np.clip((X - mu) / sd, -10, 10)

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xs[tr], y[tr])

    out = {"model_name": "Forensic-only (logistic regression)",
           "n_features": len(names), "feature_names": names, "splits": {}}
    scores = {}
    for s in ("train", "val", "logo"):
        m = sp == s
        p = clf.predict_proba(Xs[m])[:, 1]
        scores[f"{s}_p"], scores[f"{s}_y"] = p, y[m]
        out["splits"][s] = threshold_metrics(y[m], p)
        r = out["splits"][s]
        print(f"[forensic-only] {s:5s} AUC={r['auc']:.4f} AP={r['ap']:.4f} "
              f"acc@0.5={r['at_0.5']['accuracy']:.4f} n={m.sum():,}", flush=True)

    w = clf.coef_[0]
    out["weights"] = {names[i][2:]: float(w[i]) for i in range(len(names))}
    print("\n[forensic-only] top 12 features by |weight|:")
    for i in np.argsort(-np.abs(w))[:12]:
        print(f"  {names[i][2:]:24s} {w[i]:+.3f}")

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "forensic_only.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(LOGS / "forensic_only_scores.npz", **scores)


if __name__ == "__main__":
    main()

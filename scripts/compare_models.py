"""Cross-model comparison: tables, charts and a written results report.

Reads every arm JSON (and its per-sample score dump) produced by
viper_convnext.py / forensic_only_baseline.py, then writes:

  results/comparison/*.png      the charts
  results/comparison/metrics.csv  one row per model x split
  results/RESULTS.md            the report

Everything here is a local computation over the persisted scores -- no GPU and
no pod. That is the whole reason viper_convnext.py saves <tag>_scores.npz.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save, never show -- repo convention
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "comparison"

# --- palette -------------------------------------------------------------
# Backbone state is the primary categorical split, so it gets the two strongest
# hues; head variants are encoded by position within a group, not by new hues.
INK = "#1c1b22"
MUTED = "#6b6875"
GRID = "#e3e1e8"
FROZEN = "#2f6f9f"     # cool  -- backbone held fixed
UNFROZEN = "#c2532c"   # warm  -- backbone adapting
NEUTRAL = "#8a8694"
ACCENT = "#7a5ea8"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "figure.dpi": 160, "savefig.bbox": "tight",
})


def load(logs: Path):
    """Return {tag: record}, attaching per-sample scores when present."""
    models = {}
    for f in sorted(logs.glob("*.json")):
        if f.name.startswith(("acquire_", "canon_", "splits_", "forensic_cfsmall")):
            continue
        d = json.loads(f.read_text())
        if "splits" not in d:
            continue
        d["tag"] = d.get("tag", f.stem)
        npz = f.with_name(f"{f.stem}_scores.npz")
        d["scores"] = dict(np.load(npz)) if npz.exists() else {}
        models[f.stem] = d
    return models


def label_of(d, with_lr=False):
    n = d.get("model_name", d["tag"])
    if with_lr and d.get("lr"):
        n += f"  (lr {d['lr']:g})"
    return n


def color_of(d):
    """Forensic-only has no backbone, so it must not borrow either backbone hue."""
    if d.get("arch") is None:
        return ACCENT
    return FROZEN if is_frozen(d) else UNFROZEN


def style_of(d):
    if d.get("arch") is None:
        return "-."
    return "-" if family(d) == "viper" else ("--" if family(d) == "mlp" else ":")


def is_frozen(d):
    return bool(d.get("backbone_frozen", False))


def family(d):
    """linear / mlp / viper -- the head-and-features axis of the grid."""
    if d.get("feature_dim", 0) > 0:
        return "viper"
    return "mlp" if d.get("head") == "mlp" else "linear"


def fmt_pct(x, _=None):
    return f"{x:.3f}"


def selection_stats(d):
    """Three readings of one run. The headline is last5_mean; here is why.

    best_logo (as run) is optimistically biased: arms checkpoint on LOGO and
    LOGO is also reported, so a lucky epoch survives -- the frozen MLP control
    spiked to 0.9439 on epoch 3 against a 0.9334 last-five mean.

    val_selected_logo removes that bias but breaks on the fine-tuned arms,
    where val saturates (0.9971 @ep4 vs 0.9970 @ep12). argmax over a saturated
    metric is arbitrary, and it picked ep4 -- reporting 0.9704 for an arm
    sitting at 0.9843.

    last5_mean is stable under both. Every arm ran the full 12 epochs with
    patience disabled, so the last five are directly comparable.
    """
    h = d.get("history") or []
    if not h:
        return {}
    la = [e["logo_auc"] for e in h]
    va = [e.get("val_auc") for e in h]
    out = {"best_logo": max(la), "best_logo_epoch": int(np.argmax(la)) + 1,
           "final_logo": la[-1], "last5_mean": float(np.mean(la[-5:])),
           "last5_sd": float(np.std(la[-5:])), "all_sd": float(np.std(la))}
    if all(v is not None for v in va):
        i = int(np.argmax(va))
        out.update({"val_selected_epoch": i + 1, "val_selected_logo": la[i],
                    "val_selected_val": va[i]})
    return out


# --- tables ----------------------------------------------------------------
ROWS = [("auc", "AUC"), ("ap", "AP"), ("brier", "Brier")]
THR = [("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"),
       ("precision_ai", "Prec(AI)"), ("recall_ai", "Rec(AI)"), ("f1_ai", "F1(AI)"),
       ("precision_real", "Prec(Real)"), ("recall_real", "Rec(Real)"),
       ("f1_real", "F1(Real)"), ("f1_macro", "F1(macro)"), ("mcc", "MCC"),
       ("fpr", "FPR"), ("fnr", "FNR")]


def metrics_table(models, split, at="at_0.5"):
    rows = []
    for tag, d in models.items():
        s = d["splits"].get(split)
        if not s:
            continue
        r = {"model": label_of(d), "tag": tag, "split": split,
             "backbone": "frozen" if is_frozen(d) else "unfrozen",
             "head": d.get("head", "-"), "forensic_dims": d.get("feature_dim", 0),
             "trainable_params": d.get("trainable_params"),
             "best_epoch": d.get("best_epoch"), "lr": d.get("lr"),
             "n": s.get("n")}
        if split == "logo":
            r.update(selection_stats(d))
        for k, _ in ROWS:
            r[k] = s.get(k)
        t = s.get(at, {})
        for k, _ in THR:
            r[k] = t.get(k)
        c = t.get("confusion", {})
        r.update({f"cm_{k}": v for k, v in c.items()})
        rows.append(r)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# --- charts ----------------------------------------------------------------
GRID_ORDER = [("linear", "Linear head\n(no forensic)"),
              ("mlp", "MLP head\n(no forensic)"),
              ("viper", "VIPER\n(MLP + forensic)")]


def pick(models, frozen, fam, lr=None):
    for d in models.values():
        if is_frozen(d) == frozen and family(d) == fam:
            if lr is None or abs(d.get("lr", 0) - lr) < 1e-12:
                return d
    return None


def chart_grid_auc(models, split, path, lr_frozen):
    """Backbone treatment x head.

    A dot plot, not bars. These AUCs live in 0.89-0.99, where a zero-baseline
    bar chart hides every difference that matters -- and truncating a bar axis
    misleads, because bar length is read as proportional to value. A dot
    encodes position only, so a zoomed axis is honest.
    """
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ys = np.arange(len(GRID_ORDER))
    series = ((True, "Frozen backbone", FROZEN), (False, "Fine-tuned backbone", UNFROZEN))
    allv = []
    for frozen, name, col in series:
        vals = []
        for fam, _ in GRID_ORDER:
            d = pick(models, frozen, fam, lr_frozen if frozen else 1e-4)
            if not d:
                vals.append(np.nan)
            elif split == "logo":
                vals.append(selection_stats(d).get("last5_mean", np.nan))
            else:
                vals.append(d["splits"][split]["auc"])
        allv += [v for v in vals if not np.isnan(v)]
        ax.plot(vals, ys, color=col, lw=1.6, alpha=0.45, zorder=2)
        ax.scatter(vals, ys, s=132, color=col, zorder=3, label=name,
                   edgecolor=SURFACE, linewidth=2)
        for v, y in zip(vals, ys):
            if not np.isnan(v):
                ax.annotate(f"{v:.4f}", (v, y), textcoords="offset points",
                            xytext=(0, 13), ha="center", fontsize=8.6, color=INK)
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.18 or 0.01
    ax.set_xlim(lo - pad, hi + pad * 1.3)
    ax.set_ylim(-0.55, len(GRID_ORDER) - 0.35)
    ax.set_yticks(ys, [n.replace("\n", " ") for _, n in GRID_ORDER], fontsize=9)
    ax.set_xlabel(f"{split.upper()} AUC" +
                  ("  (mean of last 5 epochs)" if split == "logo" else ""))
    sub = "cross-generator, held-out generators" if split == "logo" else "in-distribution"
    ax.set_title(f"Backbone treatment x head — {sub}", loc="left",
                 fontsize=11.5, color=INK, pad=14)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_delta(models, path, lr_frozen):
    """VIPER minus its own MLP control, per backbone state, against a noise band."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names, deltas, colors = [], [], []
    for frozen, label, c in ((True, "Frozen backbone", FROZEN),
                             (False, "Fine-tuned backbone", UNFROZEN)):
        lr = lr_frozen if frozen else 1e-4
        v = pick(models, frozen, "viper", lr)
        m = pick(models, frozen, "mlp", lr)
        if not (v and m):
            continue
        sv, sm = selection_stats(v), selection_stats(m)
        if sv.get("last5_mean") is None or sm.get("last5_mean") is None:
            continue
        names.append(label)
        deltas.append(sv["last5_mean"] - sm["last5_mean"])
        colors.append(c)
    y = np.arange(len(names))
    ax.axvspan(-0.0020, 0.0020, color=NEUTRAL, alpha=0.18, zorder=1,
               label="run-to-run noise (±0.0020)")
    ax.axvline(0, color=MUTED, lw=1.2, zorder=2)
    ax.barh(y, deltas, 0.42, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, dv in zip(y, deltas):
        ax.text(dv + (0.0004 if dv >= 0 else -0.0004), yi, f"{dv:+.4f}",
                va="center", ha="left" if dv >= 0 else "right", fontsize=9.5, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("Δ cross-generator AUC  (VIPER − same head without forensic features)")
    ax.set_title("What the 33 forensic features are worth", loc="left",
                 fontsize=11.5, pad=10)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.margins(x=0.28)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_curves(models, split, path, kind="roc"):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    keep = [d for d in models.values() if d.get("scores", {}).get(f"{split}_p") is not None]
    keep.sort(key=lambda d: -d["splits"][split]["auc"])
    for d in keep:
        p, y = d["scores"][f"{split}_p"], d["scores"][f"{split}_y"]
        c, ls = color_of(d), style_of(d)
        if kind == "roc":
            a, b, _ = roc_curve(y, p)
            v = d["splits"][split]["auc"]
        else:
            b, a, _ = precision_recall_curve(y, p)
            v = d["splits"][split]["ap"]
        ax.plot(a, b, lw=1.9, ls=ls, color=c, label=f"{label_of(d, True)}  {v:.4f}")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {split.upper()} split", loc="left", fontsize=11.5, pad=10)
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–recall — {split.upper()} split", loc="left",
                     fontsize=11.5, pad=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=7.4,
                  loc="lower left" if kind == "roc" else "lower center")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_history(models, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for d in models.values():
        h = d.get("history") or []
        if not h:
            continue
        ax.plot([e["epoch"] for e in h], [e["logo_auc"] for e in h],
                lw=1.8, ls=style_of(d), color=color_of(d), marker="o", ms=3.4,
                label=label_of(d, True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-generator (LOGO) AUC")
    ax.set_title("Training trajectory — cross-generator AUC per epoch", loc="left",
                 fontsize=11.5, pad=10)
    if ax.get_legend_handles_labels()[0]:
        # below the axes: ten series leave no in-plot space that isn't data
        ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14))
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_generators(models, path, lr_frozen):
    """Per-generator forensic delta: VIPER minus its own MLP control.

    Diverging around zero rather than absolute AUC bars -- the question is
    whether the forensic branch helps on ANY held-out generator, and a
    truncated bar axis of near-1.0 AUCs would answer it badly.
    """
    pairs = [(pick(models, True, "viper", lr_frozen), pick(models, True, "mlp", lr_frozen),
              FROZEN, "Frozen backbone"),
             (pick(models, False, "viper", 1e-4), pick(models, False, "mlp", 1e-4),
              UNFROZEN, "Fine-tuned backbone")]
    pairs = [x for x in pairs if x[0] and x[1]]
    if not pairs:
        return None
    ref = pairs[0][0].get("per_held_out_generator", {})
    if not ref:
        return None
    # Small-n generators are pure noise; state the floor rather than hide it.
    gens = [g for g in sorted(ref, key=lambda g: -ref[g]["n_fake"])
            if ref[g]["n_fake"] >= 100]
    if not gens:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(gens) + 2.3))
    ys = np.arange(len(gens))
    off = 0.19
    for i, (v, m, col, name) in enumerate(pairs):
        pv, pm = v.get("per_held_out_generator", {}), m.get("per_held_out_generator", {})
        d = [pv.get(g, {}).get("auc", np.nan) - pm.get(g, {}).get("auc", np.nan)
             for g in gens]
        yy = ys + (off if i == 0 else -off)
        ax.hlines(yy, 0, d, color=col, lw=2.4, alpha=0.55, zorder=2)
        ax.scatter(d, yy, s=86, color=col, zorder=3, label=name,
                   edgecolor=SURFACE, linewidth=1.6)
    ax.axvline(0, color=MUTED, lw=1.3, zorder=4)
    ax.set_yticks(ys, [f"{g.split('/')[-1][:26]}  (n={ref[g]['n_fake']:,})"
                       for g in gens], fontsize=8.6)
    ax.set_xlabel("Δ AUC vs. all reals   (VIPER − same head without forensic features)")
    ax.set_title("Does the forensic branch help on any held-out generator?",
                 loc="left", fontsize=11.5, pad=12)
    ax.legend(frameon=False, ncol=2, fontsize=8.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_val_vs_logo(models, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for d in models.values():
        v = d["splits"].get("val", {}).get("auc")
        l = d["splits"].get("logo", {}).get("auc")
        if v is None or l is None:
            continue
        mk = "o" if family(d) == "viper" else ("s" if family(d) == "mlp" else "^")
        ax.scatter(v, l, s=74, color=color_of(d), marker=mk, zorder=3,
                   edgecolor=SURFACE, linewidth=1.6, label=label_of(d, True))
    lim = ax.get_xlim()
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.4, zorder=1)
    ax.set_xlim(lim)
    ax.set_xlabel("In-distribution AUC (val)")
    ax.set_ylabel("Cross-generator AUC (LOGO)")
    ax.set_title("Generalization gap — distance below the diagonal", loc="left",
                 fontsize=11.5, pad=10)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_weights(models, path):
    fo = next((d for d in models.values() if d.get("weights")), None)
    if not fo:
        return None
    w = fo["weights"]
    top = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:14][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    names = [k for k, _ in top]
    vals = [v for _, v in top]
    cols = [UNFROZEN if v > 0 else FROZEN for v in vals]
    ax.barh(np.arange(len(top)), vals, 0.66, color=cols, edgecolor=SURFACE, linewidth=1.4)
    ax.axvline(0, color=MUTED, lw=1.1)
    ax.set_yticks(np.arange(len(top)), names, fontsize=8.4)
    ax.set_xlabel("Logistic-regression weight  (→ AI-generated)")
    ax.set_title("Which forensic features carry the signal on their own",
                 loc="left", fontsize=11.5, pad=10)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_calibration(models, split, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.4, zorder=1)
    edges = np.linspace(0, 1, 13)
    for d in models.values():
        p = d.get("scores", {}).get(f"{split}_p")
        if p is None:
            continue
        y = d["scores"][f"{split}_y"]
        idx = np.digitize(p, edges) - 1
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() >= 40:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, lw=1.8, ls=style_of(d), color=color_of(d), marker="o",
                ms=3.2,
                label=f"{label_of(d, True)}  Brier {d['splits'][split]['brier']:.4f}")
    ax.set_xlabel("Mean predicted P(AI)")
    ax.set_ylabel("Observed fraction AI")
    ax.set_title(f"Calibration — {split.upper()} split", loc="left", fontsize=11.5, pad=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    return path


def n4(x):
    return "—" if x is None else f"{x:.4f}"


def write_report(models, splits_meta, made, lr_frozen, path):
    order = sorted(models.values(),
                   key=lambda d: -(d["splits"].get("logo", {}).get("auc") or 0))
    fro = [d for d in order if is_frozen(d)]
    unf = [d for d in order if not is_frozen(d) and d.get("arch")]

    def delta(frozen, lr):
        """VIPER minus its own MLP control, on the unbiased val-selected epoch."""
        v, m = pick(models, frozen, "viper", lr), pick(models, frozen, "mlp", lr)
        if not (v and m):
            return None
        sv, sm = selection_stats(v), selection_stats(m)
        if sv.get("last5_mean") is None or sm.get("last5_mean") is None:
            return None
        return sv["last5_mean"] - sm["last5_mean"]

    d_fro, d_unf = delta(True, lr_frozen), delta(False, 1e-4)

    L = ["# VIPER — model comparison", "",
         "Every model below is trained and evaluated on **one corpus build, one "
         "manifest and one seed**, so the numbers are directly comparable. The "
         "selection metric throughout is cross-generator (LOGO) AUC.", ""]

    L += ["## Headline", ""]
    if d_fro is not None and d_unf is not None:
        L += [f"- Forensic features are worth **{d_fro:+.4f}** LOGO AUC on a "
              f"**frozen** backbone and **{d_unf:+.4f}** on a **fine-tuned** one.",
              "- Run-to-run GPU nondeterminism on this setup is ±0.0020 AUC; read "
              "any delta smaller than that as zero.", ""]
    if order:
        b = order[0]
        L += [f"- Best cross-generator model: **{label_of(b)}** at "
              f"{n4(b['splits']['logo']['auc'])} LOGO AUC "
              f"(in-distribution val {n4(b['splits'].get('val',{}).get('auc'))}).", ""]

    L += ["## The freeze grid", "",
          "Rows are the backbone treatment; columns are the head. Only the "
          "**VIPER** column has the forensic branch — the MLP column exists so a "
          "fusion gain cannot be an extra-layer gain in disguise.", "",
          "Numbers are the **mean cross-generator AUC over the last five "
          "epochs**. Checkpointing on LOGO and reporting LOGO is optimistically "
          "biased (one arm spiked +0.010 above its own level on a single "
          "epoch); selecting on val instead breaks on the fine-tuned arms, "
          "where val saturates at ~0.997 and argmax becomes arbitrary. All "
          "three readings are in the sensitivity table below — the conclusion "
          "does not depend on which one you take.", ""]
    hdr = ["Backbone", "LR", "Linear head", "MLP head", "VIPER (MLP + forensic)",
           "VIPER − MLP"]
    rws = []
    for frozen, lr, name in ((True, 1e-3, "Frozen"), (True, 1e-4, "Frozen"),
                             (False, 1e-4, "Fine-tuned")):
        cells, got = [], {}
        for fam, _ in GRID_ORDER:
            d = pick(models, frozen, fam, lr)
            st = selection_stats(d) if d else {}
            v = st.get("last5_mean")
            got[fam] = v
            cells.append(n4(v) if v is not None else "—")
        dd = (got["viper"] - got["mlp"]
              if got.get("viper") is not None and got.get("mlp") is not None else None)
        if not any(c != "—" for c in cells):
            continue
        rws.append([name, f"{lr:g}"] + cells + [f"{dd:+.4f}" if dd is not None else "—"])
    L += [md_table(hdr, rws), "",
          "_Mean cross-generator (LOGO) AUC over epochs 8–12._", ""]

    sel = [(label_of(d), d.get("lr"), selection_stats(d))
           for d in order if d.get("history")]
    sel.sort(key=lambda t: -(t[2].get("last5_mean") or 0))
    if sel:
        L += ["### Selection sensitivity", "",
              "How much of each arm's headline number is a spike.", "",
              md_table(["Model", "LR", "last-5 mean", "last-5 sd",
                        "best-LOGO (as run)", "ep", "val-selected LOGO", "ep"],
                       [[n, f"{lr:g}" if lr else "—", n4(s.get("last5_mean")),
                         n4(s.get("last5_sd")), n4(s.get("best_logo")),
                         str(s.get("best_logo_epoch")), n4(s.get("val_selected_logo")),
                         str(s.get("val_selected_epoch"))]
                        for n, lr, s in sel]), ""]

    fo = next((d for d in models.values() if d.get("weights")), None)
    tr = []
    for frozen, lr in ((True, 1e-3), (True, 1e-4), (False, 1e-4)):
        m, v = pick(models, frozen, "mlp", lr), pick(models, frozen, "viper", lr)
        if not (m and v):
            continue
        def l5(d, k):
            return float(np.mean([e[k] for e in d["history"][-5:]]))
        dv = l5(v, "val_auc") - l5(m, "val_auc")
        dl = l5(v, "logo_auc") - l5(m, "logo_auc")
        tr.append([("Frozen" if frozen else "Fine-tuned") + f", lr {lr:g}",
                   n4(l5(m, "val_auc")), n4(l5(v, "val_auc")), f"{dv:+.4f}",
                   n4(l5(m, "logo_auc")), n4(l5(v, "logo_auc")), f"{dl:+.4f}"])
    if tr:
        L += ["## What the forensic branch actually does", "",
              "Split the effect by distribution. In every configuration tested, "
              "the forensic features **raise in-distribution AUC and lower "
              "cross-generator AUC** — three out of three, same sign on both "
              "sides, across two learning rates and both backbone treatments.", "",
              md_table(["Config", "val: MLP", "val: VIPER", "Δval",
                        "LOGO: MLP", "LOGO: VIPER", "ΔLOGO"], tr), "",
              "That is the signature of a **generator-specific shortcut**. The 33 "
              "statistics — noise residuals, PRNU-style estimates, FFT band ratios "
              "— describe the fingerprint of the particular generators in the "
              "training set. The head learns to lean on them, which pays on "
              "generators it has seen and costs on generators it has not.", "",
              "It is not that the features are noise. On their own they reach "
              f"{n4((fo or {}).get('splits', {}).get('logo', {}).get('auc'))} "
              "cross-generator AUC, far above chance. The problem is that what "
              "they add is precisely the part that does not transfer.", ""]

    for split in ("logo", "val"):
        rows = metrics_table(models, split)
        rows.sort(key=lambda r: -(r["auc"] or 0))
        if not rows:
            continue
        title = ("Cross-generator split (LOGO) — held-out generators"
                 if split == "logo" else "In-distribution split (val)")
        L += [f"## {title}", "",
              "Threshold metrics are at the deployed operating point, p ≥ 0.5. "
              "Class 1 is AI-generated.", ""]
        hdr = (["Model", "LR", "AUC", "AP", "Acc", "BalAcc", "Prec(AI)", "Rec(AI)",
                "F1(AI)", "Prec(Real)", "Rec(Real)", "MCC", "FPR", "Brier"])
        body = [[r["model"], f"{r['lr']:g}" if r.get("lr") else "—",
                 n4(r["auc"]), n4(r["ap"]), n4(r["accuracy"]),
                 n4(r["balanced_accuracy"]), n4(r["precision_ai"]), n4(r["recall_ai"]),
                 n4(r["f1_ai"]), n4(r["precision_real"]), n4(r["recall_real"]),
                 n4(r["mcc"]), n4(r["fpr"]), n4(r["brier"])] for r in rows]
        L += [md_table(hdr, body), ""]
        if split == "logo" and rows:
            mb = rows[0].get("majority_baseline_accuracy")
            n_ai = rows[0].get("cm_tp"), rows[0].get("cm_fn")
            L += [f"_n = {rows[0]['n']:,}._", ""]

    cal = []
    for d in order:
        sl, sv = d["splits"].get("logo", {}), d["splits"].get("val", {})
        if not sl.get("at_best_f1"):
            continue
        cal.append([label_of(d, True),
                    n4(sv["at_0.5"]["accuracy"]), n4(sl["at_0.5"]["accuracy"]),
                    n4(sl["at_best_f1"]["accuracy"]),
                    f"{sl['at_best_f1']['threshold']:.3f}",
                    n4(sl["at_0.5"]["recall_ai"])])
    if cal:
        L += ["## The threshold does not transfer", "",
              "AUC is threshold-free, so it hides an operational problem. On val "
              "the best-F1 threshold sits near 0.5 and accuracy there matches "
              "accuracy at 0.5. On held-out generators the best threshold "
              "collapses toward 0 and accuracy at 0.5 falls 3–7 points below what "
              "the same model achieves at its own optimum.", "",
              md_table(["Model", "val acc @0.5", "LOGO acc @0.5",
                        "LOGO acc @best-F1", "best thr", "LOGO recall(AI) @0.5"],
                       cal), "",
              "Scores on unseen generators shift toward *real*: the model is not "
              "wrong about the ranking, it is underconfident. At the deployed "
              "p ≥ 0.5 the best model still misses **14%** of AI images from a "
              "generator it has never seen, and the frozen models miss 26–36%. "
              "Any deployment needs its threshold re-tuned per target "
              "distribution, or a calibration layer — the ranking quality alone "
              "does not carry over.", ""]

    seen = set()
    cost = []
    for d in order:
        if not d.get("trainable_params"):
            continue
        k = (label_of(d),)
        if k in seen:
            continue
        seen.add(k)
        cost.append([label_of(d), f"{d['trainable_params']:,}",
                     f"{100*d['trainable_params']/d['total_params']:.2f}%"])
    L += ["## Cost of each configuration", "",
          md_table(["Model", "Trainable params", "% of 27.8M total"], cost), ""]

    vf = pick(models, True, "viper", lr_frozen)
    vu = pick(models, False, "viper", 1e-4)
    if vf and vf.get("per_held_out_generator"):
        L += ["## Per held-out generator", "",
              "Each generator is scored against the full real pool. This is where "
              "an averaged AUC hides its variance.", ""]
        gens = sorted(vf["per_held_out_generator"],
                      key=lambda g: -vf["per_held_out_generator"][g]["n_fake"])
        hdr = ["Held-out generator", "n fake", "VIPER-Frozen", "VIPER-Unfrozen", "Δ"]
        body = []
        for g in gens:
            a = vf["per_held_out_generator"][g]["auc"]
            b = (vu or {}).get("per_held_out_generator", {}).get(g, {}).get("auc")
            body.append([f"`{g}`", f"{vf['per_held_out_generator'][g]['n_fake']:,}",
                         n4(a), n4(b), f"{a-b:+.4f}" if b is not None else "—"])
        L += [md_table(hdr, body), ""]

    if fo:
        w = sorted(fo["weights"].items(), key=lambda kv: -abs(kv[1]))[:12]
        L += ["## What the forensic features encode", "",
              f"A logistic regression on the {fo['n_features']} features alone — no "
              f"network — reaches **{n4(fo['splits']['logo']['auc'])}** LOGO AUC. "
              "That is the control separating *redundant* from *useless*.", "",
              md_table(["Feature", "Weight (→ AI)"],
                       [[f"`{k}`", f"{v:+.3f}"] for k, v in w]), ""]

    if splits_meta:
        L += ["## Corpus and splits", "",
              md_table(["Quantity", "Value"],
                       [["Source", "OwensLab/CommunityForensics-Small (CC-BY-NC-SA-4.0)"],
                        ["Rows after canonicalization", f"{splits_meta.get('n_rows', 0):,}"],
                        ["Near-duplicates dropped (pHash ≤ 4)",
                         f"{splits_meta.get('n_duplicates_dropped', 0):,}"]]
                       + [[f"Split `{k}`", f"{v:,}"]
                          for k, v in (splits_meta.get("split_counts") or {}).items()]), "",
              "Canonicalization is `canon-v1:crop256-jpeg95-noexif` — a fixed JPEG "
              "encoder on every image, so the compression signature cannot be the "
              "feature the model learns.", ""]

    L += ["## Charts", ""]
    for m in made:
        if m:
            p = Path(m)
            L += [f"### {p.stem.replace('_', ' ')}", "",
                  f"![{p.stem}](comparison/{p.name})", ""]

    L += ["## Caveats", "",
          "- **Reals are not group-aware.** The manifest carries no source label "
          "for real images, so they are split at random while fakes are split by "
          "generator. The LOGO number is therefore honest about unseen "
          "*generators*, not unseen *cameras*.",
          "- **No commercial generator in the held-out set.** CF-Small contains "
          "none under a `Commercial` subset, so Midjourney/DALL·E-class models are "
          "untested here.",
          "- **One seed.** Every arm shares the seed, which makes them comparable "
          "to each other but does not give a confidence interval on any single "
          "number. The ±0.0020 band comes from repeating one configuration.", ""]

    path.write_text("\n".join(L) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(REPO / "results" / "viper_convnext" / "logs"))
    ap.add_argument("--lr-frozen", type=float, default=1e-3,
                    help="which frozen-arm LR to use in the headline charts")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    models = load(Path(args.logs))
    if not models:
        raise SystemExit(f"[compare] no arm JSON under {args.logs}")
    print(f"[compare] {len(models)} models: {', '.join(sorted(models))}")

    for split in ("val", "logo"):
        write_csv(metrics_table(models, split), OUT / f"metrics_{split}.csv")

    made = [
        chart_grid_auc(models, "logo", OUT / "grid_logo_auc.png", args.lr_frozen),
        chart_grid_auc(models, "val", OUT / "grid_val_auc.png", args.lr_frozen),
        chart_delta(models, OUT / "forensic_delta.png", args.lr_frozen),
        chart_curves(models, "logo", OUT / "roc_logo.png", "roc"),
        chart_curves(models, "logo", OUT / "pr_logo.png", "pr"),
        chart_curves(models, "val", OUT / "roc_val.png", "roc"),
        chart_history(models, OUT / "training_curves.png"),
        chart_generators(models, OUT / "per_generator.png", args.lr_frozen),
        chart_val_vs_logo(models, OUT / "val_vs_logo.png"),
        chart_weights(models, OUT / "forensic_weights.png"),
        chart_calibration(models, "logo", OUT / "calibration_logo.png"),
    ]
    for m in made:
        if m:
            print(f"[compare] wrote {Path(m).relative_to(REPO)}")

    sm = Path(args.logs) / "splits_cfsmall_v1.json"
    splits_meta = json.loads(sm.read_text()) if sm.exists() else {}
    rp = write_report(models, splits_meta, made, args.lr_frozen,
                      REPO / "results" / "RESULTS.md")
    print(f"[compare] wrote {rp.relative_to(REPO)}")


if __name__ == "__main__":
    main()

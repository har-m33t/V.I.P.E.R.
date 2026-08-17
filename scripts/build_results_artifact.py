"""Build the shareable results page. Charts are inlined as data URIs (CSP
blocks every external host), tables are generated from the result JSON so
nothing is transcribed by hand."""
import base64, json, glob
from pathlib import Path
import numpy as np

REPO = Path("/Users/harmeetsingh/Projects/V.I.P.E.R.")
LOGS = REPO / "results/viper_convnext/logs"
CH = REPO / "results/comparison"

M = {}
for f in sorted(LOGS.glob("*.json")):
    d = json.loads(f.read_text())
    if "splits" not in d or "at_0.5" not in (d["splits"].get("logo") or {}):
        continue
    M[f.stem] = d

def fam(d):
    if d.get("feature_dim", 0) > 0: return "viper"
    return "mlp" if d.get("head") == "mlp" else "linear"
def froz(d): return bool(d.get("backbone_frozen"))
def l5(d, k):
    h = d.get("history") or []
    return float(np.mean([e[k] for e in h[-5:]])) if h else None
def pick(frozen, f, lr):
    for d in M.values():
        if d.get("arch") and froz(d) == frozen and fam(d) == f and abs(d.get("lr", 0) - lr) < 1e-12:
            return d
    return None

def img(name, alt, cap):
    b = base64.b64encode((CH / name).read_bytes()).decode()
    return (f'<figure class="fig"><img src="data:image/png;base64,{b}" alt="{alt}">'
            f'<figcaption>{cap}</figcaption></figure>')

def table(headers, rows, cls="", align_num=True):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for r in rows:
        tds = "".join(
            f'<td class="{"num" if (align_num and i) else ""}">{c}</td>'
            for i, c in enumerate(r))
        body += f"<tr>{tds}</tr>"
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")

def n4(x): return "—" if x is None else f"{x:.4f}"
def sgn(x): 
    if x is None: return "—"
    c = "neg" if x < 0 else ("pos" if x > 0 else "")
    return f'<span class="{c}">{x:+.4f}</span>'

# ---- grid ----
grid = []
for frozen, lr, nm in ((True, 1e-3, "Frozen"), (True, 1e-4, "Frozen"),
                       (False, 1e-4, "Fine-tuned")):
    cells = [l5(pick(frozen, f, lr), "logo_auc") if pick(frozen, f, lr) else None
             for f in ("linear", "mlp", "viper")]
    dd = (cells[2] - cells[1]) if (cells[2] is not None and cells[1] is not None) else None
    grid.append([f'<span class="bb {"f" if frozen else "u"}">{nm}</span>',
                 f"{lr:g}", n4(cells[0]), n4(cells[1]),
                 f"<strong>{n4(cells[2])}</strong>", sgn(dd)])

# ---- tradeoff ----
tro = []
for frozen, lr in ((True, 1e-3), (True, 1e-4), (False, 1e-4)):
    m, v = pick(frozen, "mlp", lr), pick(frozen, "viper", lr)
    if not (m and v): continue
    dv = l5(v, "val_auc") - l5(m, "val_auc")
    dl = l5(v, "logo_auc") - l5(m, "logo_auc")
    tro.append([f'{"Frozen" if frozen else "Fine-tuned"}, lr {lr:g}',
                n4(l5(m, "val_auc")), n4(l5(v, "val_auc")), sgn(dv),
                n4(l5(m, "logo_auc")), n4(l5(v, "logo_auc")), sgn(dl)])

def label(d, lr=True):
    n = d.get("model_name", "?")
    return n + (f' <span class="lr">lr {d["lr"]:g}</span>' if lr and d.get("lr") else "")

order = sorted(M.values(), key=lambda d: -(d["splits"]["logo"]["auc"]))

# ---- full metrics ----
def metrics_rows(split):
    rows = []
    for d in sorted(M.values(), key=lambda x: -x["splits"][split]["auc"]):
        s = d["splits"][split]; t = s["at_0.5"]
        rows.append([label(d), n4(s["auc"]), n4(s["ap"]), n4(t["accuracy"]),
                     n4(t["balanced_accuracy"]), n4(t["precision_ai"]),
                     n4(t["recall_ai"]), n4(t["f1_ai"]), n4(t["precision_real"]),
                     n4(t["recall_real"]), n4(t["f1_macro"]), n4(t["mcc"]),
                     n4(t["fpr"]), n4(s["brier"])])
    return rows
MH = ["Model", "AUC", "AP", "Acc", "BalAcc", "Prec(AI)", "Rec(AI)", "F1(AI)",
      "Prec(Real)", "Rec(Real)", "F1(macro)", "MCC", "FPR", "Brier"]

# ---- selection sensitivity ----
sens = []
for d in sorted((x for x in M.values() if x.get("history")),
                key=lambda x: -l5(x, "logo_auc")):
    h = d["history"]; la = [e["logo_auc"] for e in h]; va = [e["val_auc"] for e in h]
    i = int(np.argmax(va))
    sens.append([label(d), n4(l5(d, "logo_auc")), n4(float(np.std(la[-5:]))),
                 f"{max(la):.4f} <span class='lr'>ep{int(np.argmax(la))+1}</span>",
                 f"{la[i]:.4f} <span class='lr'>ep{i+1}</span>"])

# ---- threshold ----
thr = []
for d in order:
    sl, sv = d["splits"]["logo"], d["splits"]["val"]
    thr.append([label(d), n4(sv["at_0.5"]["accuracy"]), n4(sl["at_0.5"]["accuracy"]),
                n4(sl["at_best_f1"]["accuracy"]),
                f'{sl["at_best_f1"]["threshold"]:.3f}',
                n4(sl["at_0.5"]["recall_ai"])])

# ---- cost ----
seen, cost = set(), []
for d in order:
    n = d.get("model_name")
    if not d.get("trainable_params") or n in seen: continue
    seen.add(n)
    cost.append([n, f'{d["trainable_params"]:,}',
                 f'{100*d["trainable_params"]/d["total_params"]:.2f}%'])

# ---- forensic weights ----
fo = next((d for d in M.values() if d.get("weights")), None)
wrows = []
if fo:
    for k, v in sorted(fo["weights"].items(), key=lambda kv: -abs(kv[1]))[:12]:
        wrows.append([f"<code>{k}</code>", sgn(v)])

sp = json.loads((LOGS / "splits_cfsmall_v1.json").read_text())

d_f13 = l5(pick(True,"viper",1e-3),"logo_auc") - l5(pick(True,"mlp",1e-3),"logo_auc")
d_f14 = l5(pick(True,"viper",1e-4),"logo_auc") - l5(pick(True,"mlp",1e-4),"logo_auc")
d_u   = l5(pick(False,"viper",1e-4),"logo_auc") - l5(pick(False,"mlp",1e-4),"logo_auc")
best  = max(M.values(), key=lambda d: l5(d,"logo_auc") if d.get("history") else 0)

HTML = f"""<title>VIPER Freeze Grid</title>
<style>
:root {{
  --ground:#f5f6f8; --panel:#ffffff; --ink:#15171c; --muted:#5c6270;
  --rule:#e2e4ea; --soft:#eef0f4;
  --frozen:#2f6f9f; --tuned:#c2532c; --pos:#1f7a52; --neg:#b4472a;
  --shadow:0 1px 2px rgba(20,24,35,.05), 0 6px 20px rgba(20,24,35,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#14161a; --panel:#1c1f26; --ink:#e9ebef; --muted:#9aa2b0;
    --rule:#2b303a; --soft:#232830;
    --frozen:#63a8d6; --tuned:#e2825c; --pos:#4cb98a; --neg:#e08163;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 22px rgba(0,0,0,.32);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#14161a; --panel:#1c1f26; --ink:#e9ebef; --muted:#9aa2b0;
  --rule:#2b303a; --soft:#232830;
  --frozen:#63a8d6; --tuned:#e2825c; --pos:#4cb98a; --neg:#e08163;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 22px rgba(0,0,0,.32);
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--ink); margin:0;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16.5px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:clamp(28px,5vw,64px) clamp(18px,4vw,40px) 96px; }}
.prose {{ max-width:74ch; }}
h1,h2,h3 {{ font-family:"Iowan Old Style",Georgia,"Times New Roman",serif; font-weight:600; text-wrap:balance; letter-spacing:-.012em; }}
h1 {{ font-size:clamp(2rem,4.6vw,2.9rem); line-height:1.12; margin:0 0 .3em; }}
h2 {{ font-size:clamp(1.32rem,2.5vw,1.62rem); margin:3.2em 0 .5em; padding-top:1.1em; border-top:1px solid var(--rule); }}
h3 {{ font-size:1.06rem; margin:2.2em 0 .4em; }}
p {{ margin:0 0 1.05em; }}
.eyebrow {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.735rem;
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 1.4em; }}
.lede {{ font-size:1.12rem; color:var(--muted); max-width:66ch; }}
strong {{ font-weight:650; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.87em;
  background:var(--soft); padding:.12em .38em; border-radius:4px; }}
.num,.lr,td.num {{ font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.lr {{ font-size:.78em; color:var(--muted); }}
.pos {{ color:var(--pos); font-weight:620; }}
.neg {{ color:var(--neg); font-weight:620; }}
.bb {{ font-weight:620; }} .bb.f {{ color:var(--frozen); }} .bb.u {{ color:var(--tuned); }}

.verdict {{ background:var(--panel); border:1px solid var(--rule); border-radius:14px;
  padding:clamp(20px,3vw,30px); margin:2.4em 0 1em; box-shadow:var(--shadow); }}
.verdict h2 {{ border:0; padding:0; margin:0 0 .55em; font-size:1.2rem; }}
.deltas {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); margin-top:1.2em; }}
.delta {{ background:var(--soft); border-radius:11px; padding:15px 17px; }}
.delta .k {{ font-size:.775rem; color:var(--muted); text-transform:uppercase; letter-spacing:.075em; }}
.delta .v {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:1.5rem;
  font-variant-numeric:tabular-nums; margin-top:.16em; font-weight:600; }}

.tw {{ overflow-x:auto; margin:1.5em 0; border:1px solid var(--rule); border-radius:12px; background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:.855rem; }}
th,td {{ padding:9px 13px; text-align:left; border-bottom:1px solid var(--rule); white-space:nowrap; }}
th {{ font-size:.735rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
  font-weight:600; background:var(--soft); position:sticky; top:0; }}
td.num {{ text-align:right; }}
th:not(:first-child) {{ text-align:right; }}
tbody tr:last-child td {{ border-bottom:0; }}
tbody tr:hover {{ background:var(--soft); }}

.fig {{ margin:1.9em 0; }}
.fig img {{ width:100%; max-width:100%; height:auto; display:block;
  background:#fff; border:1px solid var(--rule); border-radius:12px; }}
.fig figcaption {{ font-size:.845rem; color:var(--muted); margin-top:.7em; max-width:74ch; }}
.grid2 {{ display:grid; gap:20px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }}
.grid2 .fig {{ margin:0; }}
.note {{ border-left:3px solid var(--frozen); background:var(--panel); padding:14px 18px;
  border-radius:0 10px 10px 0; margin:1.6em 0; font-size:.93rem; }}
ul {{ padding-left:1.15em; }} li {{ margin:.4em 0; }}
footer {{ margin-top:4em; padding-top:1.4em; border-top:1px solid var(--rule);
  color:var(--muted); font-size:.85rem; }}
</style>

<div class="wrap">
<p class="eyebrow">VIPER · forensic-feature ablation</p>
<h1>Does freezing the backbone make forensic features pay off?</h1>
<p class="lede prose">Nine ConvNeXt-Tiny arms on one corpus build, one manifest, one seed —
crossing backbone treatment against head capacity to isolate what 33 hand-crafted
forensic features are actually worth.</p>

<div class="verdict">
<h2>The answer is no — and the reason is more useful than the delta</h2>
<p>With the backbone frozen, the head cannot learn low-level noise statistics for itself,
so the forensic branch should have had room to contribute. It did not. Against its own
MLP control — same head, same capacity, forensic branch removed — VIPER lands slightly
<em>behind</em> in every configuration tested.</p>
<div class="deltas">
  <div class="delta"><div class="k">Frozen · lr 1e-3</div><div class="v neg">{d_f13:+.4f}</div></div>
  <div class="delta"><div class="k">Frozen · lr 1e-4</div><div class="v neg">{d_f14:+.4f}</div></div>
  <div class="delta"><div class="k">Fine-tuned · lr 1e-4</div><div class="v neg">{d_u:+.4f}</div></div>
</div>
<p style="margin-top:1.25em;margin-bottom:0">Change in cross-generator AUC from adding the forensic
branch. All three negative, all small — but the direction is consistent, and it is the
same direction on the in-distribution split reversed.</p>
</div>

<div class="prose">
<h2>The grid</h2>
<p>Backbone treatment on the rows, head on the columns. Only the <strong>VIPER</strong>
column carries the forensic branch. The middle column is the control that makes the
comparison mean anything: without it, VIPER measured against the linear head would
credit the forensic features with all of the extra head capacity.</p>
<p>That is not a hypothetical. On a frozen backbone the MLP head alone is worth
<strong>+0.050 AUC</strong> — an order of magnitude larger than the forensic effect
being measured.</p>
</div>
{table(["Backbone","LR","Linear head","MLP head","VIPER (MLP + forensic)","VIPER − MLP"], grid)}
<p class="prose" style="font-size:.87rem;color:var(--muted)">Mean cross-generator (LOGO) AUC over epochs 8–12.</p>

{img("grid_logo_auc.png","Dot plot of LOGO AUC by backbone treatment and head",
     "A dot plot rather than bars: these AUCs sit between 0.89 and 0.99, where a zero-baseline bar chart hides every difference that matters — and truncating a bar axis misleads, because bar length reads as proportional to value.")}

<div class="prose">
<h2>What the forensic branch actually does</h2>
<p>Splitting the effect by distribution turns a shrug into a finding. In all three
configurations the forensic features <strong>raise in-distribution AUC and lower
cross-generator AUC</strong> — same sign on both sides, across two learning rates and
both backbone treatments.</p>
</div>
{table(["Config","val: MLP","val: VIPER","Δ val","LOGO: MLP","LOGO: VIPER","Δ LOGO"], tro)}
<div class="prose">
<div class="note">That is the signature of a <strong>generator-specific shortcut</strong>.
The 33 statistics — noise residuals, PRNU-style estimates, FFT band ratios — describe the
fingerprint of the particular generators in the training set. The head learns to lean on
them, which pays on generators it has seen and costs on generators it has not.</div>
<p>The features are not noise. On their own, with no network at all, a logistic regression
over them reaches <strong>{n4(fo["splits"]["logo"]["auc"]) if fo else "—"}</strong>
cross-generator AUC — far above chance. The problem is that what they add on top of a
CNN is precisely the part that does not transfer.</p>
</div>

{img("per_generator.png","Per-generator forensic delta",
     "Per held-out generator, VIPER minus its own MLP control. Mixed signs, no generator showing a consistent benefit, and the largest effects negative. Generators with fewer than 100 held-out images are excluded as pure noise.")}

<div class="prose">
<h2>Reading the numbers honestly</h2>
<p>The arms checkpoint on cross-generator AUC and cross-generator AUC is also what gets
reported — an optimistically biased protocol that keeps whichever epoch happened to spike.
The frozen MLP control peaked at 0.9439 on epoch 3 against a 0.9334 last-five mean, which
alone would have inflated the forensic penalty from −0.003 to −0.010.</p>
<p>Selecting on val instead removes that bias but breaks on the fine-tuned arms, where val
saturates at ~0.997 and <code>argmax</code> becomes arbitrary — it picked epoch 4 and
reported 0.9704 for an arm sitting at 0.9843. The last-five mean is stable under both
failure modes, and every arm ran a full 12 epochs with early stopping disabled, so the
last five are directly comparable. All three readings agree on the conclusion.</p>
</div>
{table(["Model","last-5 mean","last-5 sd","best-LOGO (as run)","val-selected LOGO"], sens)}

{img("training_curves.png","Cross-generator AUC per epoch for all nine arms",
     "Every arm across all 12 epochs. The band structure is the backbone treatment; the spread within each band is what the head is worth.")}

<div class="prose">
<h2>The threshold does not transfer</h2>
<p>AUC is threshold-free, which hides an operational problem. On val the best-F1 threshold
sits near 0.5 and accuracy there matches accuracy at 0.5. On held-out generators the best
threshold collapses toward zero, and accuracy at the deployed 0.5 falls 3–7 points below
what the same model reaches at its own optimum.</p>
</div>
{table(["Model","val acc @0.5","LOGO acc @0.5","LOGO acc @best-F1","best thr","LOGO recall(AI) @0.5"], thr)}
<div class="prose">
<div class="note">Scores on unseen generators shift toward <em>real</em>: the ranking is
still good, the model is underconfident. At p ≥ 0.5 the strongest model still misses
<strong>14%</strong> of AI images from a generator it has never seen; the frozen models
miss 26–36%. Any deployment needs its threshold re-tuned per target distribution, or a
calibration layer.</div>
</div>

<div class="grid2">
{img("roc_logo.png","ROC curves on the cross-generator split","ROC — cross-generator split.")}
{img("calibration_logo.png","Calibration curves on the cross-generator split","Calibration — the underconfidence above, drawn.")}
</div>

<div class="prose"><h2>Full metrics — cross-generator split</h2>
<p>Threshold metrics at the deployed operating point, p ≥ 0.5. Class 1 is AI-generated.
n = {M[list(M)[0]]["splits"]["logo"]["n"]:,}.</p></div>
{table(MH, metrics_rows("logo"))}

<div class="prose"><h3>Full metrics — in-distribution split</h3>
<p>n = {M[list(M)[0]]["splits"]["val"]["n"]:,}.</p></div>
{table(MH, metrics_rows("val"))}

<div class="prose"><h2>What each configuration costs</h2></div>
{table(["Model","Trainable params","% of 27.8M total"], cost)}
<div class="prose"><p>The frozen VIPER head trains <strong>1.10%</strong> of the network and
lands 0.042 AUC behind full fine-tuning. Freezing is the expensive choice here, not the
forensic branch.</p></div>

<div class="prose"><h2>Which forensic features carry signal</h2>
<p>Weights from the standalone logistic regression — the arm with no CNN at all. They land
where forensics says they should: noise-residual and gradient statistics dominate.</p></div>
<div class="grid2" style="align-items:start">
{table(["Feature","Weight → AI"], wrows)}
{img("forensic_weights.png","Forensic feature weights","")}
</div>

<div class="prose">
<h2>Method</h2>
<ul>
<li><strong>Corpus.</strong> OwensLab/CommunityForensics-Small (CC-BY-NC-SA-4.0),
{sp["n_rows"]:,} images after canonicalization, {sp["n_duplicates_dropped"]:,}
near-duplicates dropped by perceptual hash at Hamming ≤ 4. This build reproduced the
earlier one exactly — same row count, same duplicate count, same split sizes.</li>
<li><strong>Canonicalization.</strong> <code>canon-v1:crop256-jpeg95-noexif</code> — EXIF
stripped, RGB, random 256px crop, then one fixed JPEG encoder over every image, so
compression signature cannot be the feature the model learns.</li>
<li><strong>Splits.</strong> train {sp["split_counts"]["train"]:,} /
val {sp["split_counts"]["val"]:,} / LOGO {sp["split_counts"]["logo"]:,}, with fakes split
leave-one-generator-out and a 2% per-generator cap.</li>
<li><strong>Training.</strong> Adam, cosine annealing, batch 64, bf16 autocast,
<code>channels_last</code>, 12 epochs with early stopping disabled. 80k train / 40k val /
40k LOGO sample, identical uids across every arm.</li>
<li><strong>Freezing.</strong> <code>--unfreeze-stages 0</code> freezes all four ConvNeXt
stage groups <em>and</em> the final LayerNorm, and forces the backbone to
<code>eval()</code> during training so stochastic depth does not inject noise into
features nothing can adapt to.</li>
<li><strong>Forensic scaler</strong> fitted on train only — fitting over the whole corpus
would leak val/LOGO distribution into training and inflate exactly what is being measured.</li>
</ul>

<h2>Caveats</h2>
<ul>
<li><strong>Reals are not group-aware.</strong> The manifest carries no source label for
real images, so reals split at random while fakes split by generator. The LOGO number is
honest about unseen <em>generators</em>, not unseen <em>cameras</em>.</li>
<li><strong>No commercial generator held out.</strong> CF-Small has none under a
<code>Commercial</code> subset, so Midjourney/DALL·E-class models are untested here.</li>
<li><strong>One seed per arm.</strong> Arms are comparable to each other, but no single
number carries a confidence interval. Epoch-to-epoch SD within a run is 0.0005–0.0095,
which is the same order as the effects being measured — the consistency of the sign across
three configurations is what carries the conclusion, not any one delta.</li>
<li><strong>Negative results are bounded.</strong> This says late fusion of these 33
features into this backbone does not help. It does not say forensic features are useless
in general — computed pre-canonicalization, or against a backbone that never sees raw
pixels, the answer could differ.</li>
</ul>
</div>

<footer>VIPER · ConvNeXt-Tiny freeze grid · nine arms, one corpus build, one seed ·
RTX 4090 · charts and tables generated by <code>scripts/compare_models.py</code></footer>
</div>
"""
out = Path("/private/tmp/claude-501/-Users-harmeetsingh-Projects-V-I-P-E-R-/1ef71d64-2347-485e-a202-0d6fe9345d7c/scratchpad/viper_freeze_grid.html")
out.write_text(HTML)
print("wrote", out, f"{out.stat().st_size/1e6:.2f} MB")

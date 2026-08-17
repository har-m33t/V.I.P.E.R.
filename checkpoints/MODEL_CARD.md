# VIPER — model card

Binary classifiers separating AI-generated images from real photographs, plus
the forensic-feature ablations around them.

## Naming

```
VIPER-<backbone>-<Frozen|Unfrozen>-<head>
```

The `VIPER` prefix is reserved for configurations that **fuse forensic features
with a backbone**. A model without that branch is a backbone baseline and is
named as one — `ConvNeXt-Frozen-MLP (no forensic)` — so the two never get
conflated in a results table.

- **backbone** — the feature extractor (`ConvNeXt`, `CLIP`)
- **Frozen / Unfrozen** — whether backbone weights receive gradients. `Frozen`
  freezes all four ConvNeXt stage groups *and* the final LayerNorm, and holds
  the backbone in `eval()` during training so stochastic depth does not inject
  noise into features nothing can adapt to.
- **head** — `Linear` (`Dropout → Linear(768, 2)`) or `MLP`
  (`Dropout → Linear(768+d, 384) → ReLU → Linear(384, 2)`)

`VIPERConvNeXt.variant_name()` in `src/model.py` derives this string from the
live configuration, so the name cannot drift from the model it describes.

## Architecture

`VIPERConvNeXt` wraps `torchvision.models.convnext_tiny` (`IMAGENET1K_V1`),
27,821,666 total parameters. It exposes a 768-d embedding; with
`eda_feature_dim > 0` that embedding is concatenated with the standardized
forensic vector and routed through the MLP head.

The 33 forensic features come from `src/eda.py`: pixel statistics, FFT
high/low-frequency band energy, colour-palette entropy with LAB saturation,
noise residuals with a PRNU-style estimator, GLCM texture, and Canny edge
density.

## Weights

Published as **GitHub Release assets** — every checkpoint exceeds GitHub's hard
100 MB per-file limit, so none are committed to the repository.

Verify any download against `checkpoints/SHA256SUMS`:

```bash
shasum -a 256 -c checkpoints/SHA256SUMS --ignore-missing
```

### Available

| File | Variant | Corpus | LOGO AUC | val AUC |
|---|---|---|---|---|
| `best_model.pth` | `VIPER-ConvNeXt-Unfrozen-Linear` | 12.9k art corpus | — | 0.9744 |
| `unfrozen_linear_lr1e4.pt` | `ConvNeXt-Unfrozen-Linear (no forensic)` | CF-Small | 0.9843 | 0.9967 |
| `viper_convnext_unfrozen_lr1e4.pt` | `VIPER-ConvNeXt-Unfrozen-MLP` | CF-Small | 0.9804 | 0.9975 |
| `viper_convnext_unfrozen_lr1e4_patience4.pt` | `VIPER-ConvNeXt-Unfrozen-MLP` (early-stopped) | CF-Small | 0.9822 | 0.9949 |
| `convnext_cfsmall_v1.pt` | `ConvNeXt-Unfrozen-Linear (no forensic)`, first pilot | CF-Small | 0.9854 | 0.9961 |
| `probe_clean_v1.pt`, `probe_clean_v2.pt` | CLIP ViT-L/14 linear probe (crop / resize) | CF-Small | 0.9269 / 0.9015 | 0.9597 / 0.9677 |
| `probe_blur3_v1.pt`, `probe_blur3_v2.pt` | CLIP probe, Gaussian blur-3 leakage control | CF-Small | 0.7336 / 0.7660 | 0.8621 / 0.8836 |

### Not retained

The seven frozen-backbone grid checkpoints were lost when the GPU pod was
terminated; only their logs and per-sample scores were pulled. **Every metric
they produced is preserved** in `results/viper_convnext/logs/` — per-epoch
history, threshold metrics at two operating points, and per-sample scores —
so nothing in the published results is unverifiable. The weights themselves
need a retrain:

```bash
# ~13 min per arm on one RTX 4090
python scripts/viper_convnext.py --proc-name community_forensics_small \
  --manifest cfsmall_v1 --tag viper_convnext_frozen_lr1e4 \
  --epochs 12 --patience 12 --unfreeze-stages 0 --lr 1e-4 \
  --features /data/features/cfsmall_v1
```

Affected tags: `frozen_linear_lr1e3`, `frozen_linear_lr1e4`,
`frozen_mlp_lr1e3`, `frozen_mlp_lr1e4`, `unfrozen_mlp_lr1e4`,
`viper_convnext_frozen_lr1e3`, `viper_convnext_frozen_lr1e4`.

## Training data

[OwensLab/CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
— Park & Owens, *Community Forensics*, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125)).
**Licensed CC-BY-NC-SA-4.0: non-commercial research use only.** Weights derived
from it inherit that restriction.

556,541 images after canonicalization, 1,610 near-duplicates dropped by
perceptual hash at Hamming ≤ 4. Fakes are split leave-one-generator-out; a 2%
per-generator cap prevents any single generator dominating.

## Intended use and limits

Research into generated-image detection and forensic-feature attribution.

**Not suitable as a deployed authenticity check without recalibration.** The
decision threshold does not transfer across generators: on validation the
best-F1 threshold sits near 0.5, but on held-out generators it collapses toward
0 and accuracy at 0.5 drops 3–7 points. At p ≥ 0.5 the strongest model still
misses 14% of AI images from an unseen generator; frozen variants miss 26–36%.

Further limits:

- **Reals are not group-aware.** No source label exists for real images, so
  they split at random while fakes split by generator. Cross-generator numbers
  are honest about unseen *generators*, not unseen *cameras*.
- **No commercial generator held out.** CF-Small carries none under a
  `Commercial` subset, so Midjourney/DALL·E-class models are untested.
- **One seed per arm.** Arms are comparable to each other; no single number
  carries a confidence interval.
- Images are canonicalized to 256px with a fixed JPEG q95 encoder. Behaviour on
  other resolutions or compression settings is uncharacterized.

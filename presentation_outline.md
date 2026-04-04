# VIPER Forensic Engine — Presentation Structure
# Heist 2026 Datathon — ArtHeist Challenge
# ============================================================
# 8-Slide Narrative Arc: "The Museum Director Test"
# Each technical result is paired with a plain-English sentence.
# ============================================================

---

## SLIDE 1 — The Heist Setup

**Headline:** "Every forgery leaves a fingerprint. Every AI a signature."

**Plain English:** Real paintings carry the chaos of the physical world.
AI images carry the mathematical patterns of their training data.

> Narrative beat: Set the stakes — art forgery has gone digital.

**Placeholders:**
- [RESULT] n_real vs n_ai images in dataset
- [IMAGE] dramatic split of real vs AI art example

---

## SLIDE 2 — The VIPER Engine

**Headline:** "A multi-stage forensic pipeline for the digital age."

**Plain English:** We built six forensic tests — the same tests a
crime lab would apply — before teaching a neural network to see what we see.

> Narrative beat: Introduce the pipeline architecture.

**Placeholders:**
- [DIAGRAM] Pipeline flowchart (Data → EDA → Baseline → DL → Eval → Viz)

---

## SLIDE 3 — Forensic Fingerprints (EDA)

**Headline:** "AI images are too smooth, too organized, too perfect."

**Plain English:** Real photos have natural noise. AI images have
characteristic frequency patterns, cleaner textures, and different colour palettes.

> Narrative beat: Show the six forensic features.

**Placeholders:**
- [RESULT] eda_distributions.png — feature separation between classes
- [RESULT] eda_fft_ratio.png — FFT high-frequency energy by class
- [STAT] Top-2 most discriminative features from baseline_metrics.json

---

## SLIDE 4 — The Baseline Detective

**Headline:** "Even a simple model learns the forensic fingerprint."

**Plain English:** A logistic regression on six hand-crafted features
already achieves meaningful separation — proof the signal is real.

> Narrative beat: Validate the EDA with classical ML.

**Placeholders:**
- [STAT] Baseline accuracy, F1, AUC-ROC from baseline_metrics.json
- [RESULT] Top-3 features by coefficient

---

## SLIDE 5 — The Neural Detective

**Headline:** "EfficientNet-B0 sees what the eye cannot."

**Plain English:** A fine-tuned deep network learns to combine thousands
of subtle patterns — far beyond what hand-crafted features can capture.

> Narrative beat: Present the DL results.

**Placeholders:**
- [RESULT] Accuracy, F1, Precision, Recall, AUC-ROC from eval_metrics.json
- [RESULT] confusion_matrix.png
- [COMPARE] Baseline accuracy vs EfficientNet accuracy (delta)

---

## SLIDE 6 — Reading the Model's Mind (Grad-CAM)

**Headline:** "The model flags the brush strokes, not the canvas."

**Plain English:** Grad-CAM shows us exactly which regions drove the
decision — AI images are often caught by their hair, skin tone perfection,
or uniform background gradients.

> Narrative beat: Explainability — trust the model.

**Placeholders:**
- [IMAGE] 4-panel grid from gradcam_gallery/ (2 correct, 2 misclassified)

---

## SLIDE 7 — The Evidence Room (UMAP)

**Headline:** "Real and AI art live in separate universes."

**Plain English:** When we map 1280 model features down to two dimensions,
real and AI images cluster apart — the neural network learned a
genuine forensic boundary.

> Narrative beat: Structural separation = real discrimination, not memorization.

**Placeholders:**
- [RESULT] omni_export/umap_scatter.png
- [RESULT] wikiart_confidence.json — how WikiArt art scores

---

## SLIDE 8 — Verdict & Future Work

**Headline:** "The VIPER Engine convicted [X]% of AI forgeries."

**Plain English:** Our engine correctly identifies AI-generated images
with high confidence. Next: real-time browser extension, style attribution,
and generative model fingerprinting.

> Narrative beat: Close with impact + what's next.

**Placeholders:**
- [STAT] Final AUC-ROC from eval_metrics.json
- [FUTURE] Three next-step bullet points:
    1. Extend to video deepfakes
    2. Train on latest diffusion model outputs (SDXL, FLUX)
    3. Deploy as an API / browser extension for public use

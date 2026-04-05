# 🏁 VIPER Forensic Engine — Final Dashboard & Integration Agents

This document contains the final mission-critical agent prompts and task lists required to complete the Heist 2026 Datathon deliverable. The goal is to bind our high-performing backend into a flawless, wildly impressive Streamlit Dashboard.

To deploy an agent, copy the **"System Prompt"** and give it to an AI, followed by the **"Next Tasks"** list.

---

## 🖥️ 1. The Lead Streamlit Architect
**Target Files:** `app.py` (New), `requirements.txt`

### System Prompt
> You are a Lead Frontend Data App Developer specializing in `Streamlit`. Your objective is to build the central `app.py` dashboard that serves as the final presentation layer for the VIPER Forensic Engine. The judges must be wowed by the UI polish.
> 
> **Your Constraints:**
> - Build a 2-view structure (e.g., using `st.tabs` or `st.sidebar.radio`): "Image Forensics" and "Model Intelligence".
> - You must use caching (`@st.cache_data`, `@st.cache_resource`) to ensure changing tabs or uploading images feels instant (< 1–2s).
> - Make key outputs visually prominent using Streamlit columns, metrics, and expanders for secondary info.

### Next Tasks to Execute
- [x] **Task 1: The Core Structure**: Initialize `app.py` and set up the two main views ("Image Forensics" and "Model Intelligence View") with a clean sidebar navigation. Apply a dark-themed CSS injection if possible to maintain the cybersecurity aesthetic.
- [x] **Task 2: Image Forensics View (Upload)**: Implement a file uploader or sample selector. Display the original image side-by-side with the Grad-CAM overlay. Collapse any heavy technical plots (FFT, PRNU, LAB) into a clean `st.expander` (Note: EDA features are currently disabled for the 10k fast-track, so rely heavily on Deep Learning Confidence metrics).
- [x] **Task 3: Model Intelligence View**: Integrate the pre-existing UMAP scatter plots and create placeholders for the Fusion Comparison chart and Error Breakdown charts.

---

## 🔬 2. The Forensic Signal & Visualization Agent
**Target Files:** `src/eda.py`, `src/visualize.py`

### System Prompt
> You are a Senior Data Visualization and Forensic Signal Engineer. Your objective is to map our raw metrics into human-readable buckets and generate the final holistic presentation plots.
>
> **Your Constraints:**
> - You own `src/eda.py` (for signal scoring) and `src/visualize.py` (for plotting).
> - Standardize all EDA outputs to return a strict quantitative score and a label: `{ "score": float, "label": "low/medium/high" }`.
> - All new plots must be returned as Plotly or Matplotlib objects so the Streamlit agent can render them easily.

### Next Tasks to Execute
- [x] **Task 1: Signal Scoring Layer**: Update `eda.py` to bucket FFT irregularities, PRNU noise variations, and LAB saturation into "Low / Medium / High" anomaly scores based on standard deviations from the dataset mean.
- [x] **Task 2: Forensic Report Object**: Create a strict dictionary generator that takes the image prediction, confidence, and the 3 EDA bucket scores, and calculates a rule-based "Final Verdict" string (e.g., "Strong AI likelihood").
- [x] **Task 3: Confidence Heatmap (2D UMAP)**: Integrate the new, optimized 2D interactive Plotly UMAP scatter. Add a toggle (or separate plot) isolating only the misclassified edge-cases to show where the model struggles.
- [x] **Task 4: Fast-Track Comparison Visual**: Collect the F1 baseline metrics and the new 10k-Fine-Tuned metrics from `eval_metrics.json`. Build a Plotly Bar Chart comparing the two to highlight the performance jump achieved by the 67k dataset scale-up.

---

## 🗣️ 3. The 'Omni Lite' Explainer Agent
**Target Files:** `src/omni.py` (New)

### System Prompt
> You are an AI Interpretability Communicator. Your job is to translate complex math and forensic matrices into concise, non-technical English sentences that a business executive or datathon judge can instantly understand.
>
> **Your Constraints:**
> - You will create functions in `src/omni.py`.
> - Your outputs must be 1-2 sentence strings with a highly consistent, professional tone.

### Next Tasks to Execute
- [x] **Task 1: The Report Explainer**: Write a function that takes the `Forensic Report Card` dictionary and returns a 1-2 sentence human-readable insight (e.g., *"The model is 92% confident this is AI-generated, leaning heavily on unnatural color saturation and high-frequency FFT anomalies."*)
- [x] **Task 2: Error Insight Generator**: Create a function that takes the Error Breakdown stats (e.g., false positives) and generates a 1 sentence observation on the model's blind spots.

---

## ⚙️ 4. The Data Pipeline & Optimization Agent
**Target Files:** `src/dataloader.py`, `app.py`

### System Prompt
> You are an MLOps and Performance Optimization Engineer. Your sole purpose is to ensure the Streamlit dashboard never lags during the live demo and that the data flowing into it is 100% clean and consistent.
>
> **Your Constraints:**
> - No data leakage (do not mix test and train data in the presentation visuals).
> - Image inference must happen in < 1-2 seconds.

### Next Tasks to Execute
- [x] **Task 1: Precomputation Engine**: Write a script to precompute all predictions, confidences, and 768D embeddings for the 10k fast-track validation set and save them to JSON/CSV files so the dashboard boots instantly without running PyTorch.
- [x] **Task 2: Dashboard State Optimization**: Review `app.py` alongside the Streamlit Agent and wrap all model loading and heavy UMAP calculations in `@st.cache_resource` decorators. Avoid recomputing signals unnecessarily if an image is just toggled.
- [x] **Task 3: Error Breakdown Analysis**: Identify False Positives and False Negatives from the predicted json output. Group them by confidence buckets (low/high) and generate a bar/pie chart counting the frequency of each failure mode.

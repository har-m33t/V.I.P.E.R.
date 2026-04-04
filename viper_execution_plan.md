# VIPER Forensic Engine — Concurrency Execution Plan

## Directory Manifest
- `data/cifake/` — Directory containing Kaggle Ai Art vs Real Art dataset
- `data/wikiart/` — Directory containing 1K WikiArt holdout dataset.
- `src/dataloader.py` — Script for data ingestion, validation, and batching. Owned by: Data Agent.
- `src/eda.py` — Script to compute the six required EDA metrics. Owned by: EDA Agent.
- `results/feature_matrix.csv` — Matrix containing image paths and extracted EDA features. Owned by: EDA Agent.
- `src/baseline.py` — Script for Logistic Regression baseline. Owned by: Baseline Agent.
- `results/baseline_metrics.json` — Baseline Logistic Regression performance and top-3 features. Owned by: Baseline Agent.
- `src/model.py` — EfficientNet-B0 network definition and configuration. Owned by: Deep Learning Agent.
- `src/train.py` — Training loop with CosineAnnealingLR and checkpointing. Owned by: Deep Learning Agent.
- `checkpoints/best_model.pth` — Saved model checkpoint based on validation F1. Owned by: Deep Learning Agent.
- `src/evaluate.py` — Script calculating core metrics and UMAP embeddings. Owned by: Evaluation Agent.
- `results/eval_metrics.json` — Accuracy, F1, precision, recall, AUC-ROC scores. Owned by: Evaluation Agent.
- `results/confusion_matrix.png` — Plotted confusion matrix. Owned by: Evaluation Agent.
- `results/umap_features.csv` — Extracted UMAP features of final-layer embeddings. Owned by: Evaluation Agent.
- `src/visualize.py` — Generates Grad-CAM galleries and final UMAP plots. Owned by: Visualization Agent.
- `gradcam_gallery/` — Directory containing 20 correct classification and 5 misclassification images. Owned by: Visualization Agent.
- `omni_export/umap_scatter.png` — Main UMAP deliverable for the Omni data track. Owned by: Visualization Agent.
- `omni_export/metadata.csv` — Omni metadata linking UMAP points to Grad-CAM results. Owned by: Visualization Agent.
- `src/stretch.py` — Runs JPEG robustness and WikiArt patch tasks. Owned by: Stretch Agent.
- `results/jpeg_robustness.png` — Plot of accuracy vs quality curve indicating degradation cliff. Owned by: Stretch Agent.
- `results/wikiart_confidence.json` — Confidence distribution for WikiArt inference. Owned by: Stretch Agent.
- `notebooks/assembly.ipynb` — Master notebook threading results together for submission. Owned by: Notebook Agent.
- `presentation_outline.md` — 8-slide deck text structure for presentation. Owned by: Presentation Agent.
- `README.md` — Final documentation and repository map. Owned by: Documentation Agent.

## Parallel Execution Tracks

To maximize speed, agents are divided into autonomous execution tracks. Tracks Beta, Gamma, and Epsilon operate simultaneously once Track Alpha is complete. Internally, each track executes sequentially.

*   **Track Alpha: Foundation**
    1.  **Data Agent**: Provisions datasets and builds `src/dataloader.py`. 
*   **Track Beta: Analytical Pipeline** (Runs parallel to Gamma & Epsilon)
    1.  **EDA Agent**: Waits for Track Alpha. Computes `results/feature_matrix.csv`.
    2.  **Baseline Agent**: Waits for EDA Agent. Computes `results/baseline_metrics.json`.
*   **Track Gamma: Deep Learning Pipeline** (Runs parallel to Beta & Epsilon)
    1.  **Deep Learning Agent**: Waits for Track Alpha. Trains EfficientNet, outputs `checkpoints/best_model.pth`.
    2.  **Evaluation Agent**: Waits for Deep Learning Agent. Computes core metrics and UMAP embeddings.
    3.  **Visualization Agent**: Waits for Evaluation Agent. Generates Grad-CAM galleries and Omni track exports.
*   **Track Delta: Stretch Pipeline** (Runs parallel to Gamma's Evaluation/Visualization)
    1.  **Stretch Agent**: Waits for Deep Learning Agent. Computes jpeg robustness and WikiArt inference independently.
*   **Track Epsilon: Communications** (Runs concurrently with ALL tracks)
    1.  **Presentation Agent**: Starts immediately. Drafts the 8-slide Heist structured outline.
    2.  **Documentation Agent**: Starts immediately. Drafts the architecture sections of the README.
    3.  **Notebook Agent**: Waits for all Tracks to produce their final output flags. Assembles the final master notebook.

## Agent Task Briefs

### Data Agent (Track Alpha)
You are the Data Agent. You own `src/dataloader.py`. You are the first agent to run. You must validate the Kaggle CIFAKE and WikiArt datasets. If they do not exist in `data/cifake/` and `data/wikiart/`, your script must crash and report the error. You must construct PyTorch DataLoaders with batch size 64 for binary classification. If you time out, you must output an empty `src/dataloader.py` that raises a `NotImplementedError` so downstream tracks do not hang.

### EDA Agent (Track Beta)
You are the EDA Agent. You own `src/eda.py` and `results/feature_matrix.csv`. You may not begin until `src/dataloader.py` exists on disk (Track Alpha complete). You must compute exactly six features across the dataset: pixel statistics, FFT frequency analysis, color palette (K-Means k=8), noise residuals, GLCM texture metrics, and Canny edge density. You must output these figures and feature values directly to `results/feature_matrix.csv`. If you stall, calculate only pixel statistics and output the partial `results/feature_matrix.csv`.

### Baseline Agent (Track Beta)
You are the Baseline Agent. You own `src/baseline.py` and `results/baseline_metrics.json`. You may not begin until `results/feature_matrix.csv` exists on disk. You must implement a Logistic Regression baseline model reading exactly from that CSV. You must output performance metrics to `results/baseline_metrics.json` and print the top-3 predictive features by coefficient. If you fail to finish, write a dummy `results/baseline_metrics.json` containing `{"status": "failed"}`.

### Deep Learning Agent (Track Gamma)
You are the Deep Learning Agent. You own `src/model.py`, `src/train.py`, and `checkpoints/best_model.pth`. You may not begin until `src/dataloader.py` exists (Track Alpha complete). You must implement an EfficientNet-B0 architecture. You must freeze the base, and unfreeze only the last 3 MBConv blocks plus the classifier. Use Adam optimizer with lr=1e-4, CosineAnnealingLR scheduling, exactly 10 epochs, and a batch size of 64. Save the best-performing model by validation F1 score to `checkpoints/best_model.pth`. If you process times out, save whatever checkpoint you have to `checkpoints/best_model.pth`.

### Evaluation Agent (Track Gamma)
You are the Evaluation Agent. You own `src/evaluate.py`. You may not begin until `checkpoints/best_model.pth` exists on disk. You must evaluate the holdout test set through the model. Calculate accuracy, F1, precision, recall, and AUC-ROC, and save them exactly to `results/eval_metrics.json`. Generate a confusion matrix and save it to `results/confusion_matrix.png`. Extract final-layer embeddings and reduce them using UMAP, saving the 2D coordinates to `results/umap_features.csv`. If you time out, write `{"accuracy": 0}` to `results/eval_metrics.json` and create dummy files for the rest.

### Visualization Agent (Track Gamma)
You are the Visualization Agent. You own `src/visualize.py`. You may not begin until `results/eval_metrics.json` and `results/umap_features.csv` exist on disk. Generate Grad-CAM heatmaps for exactly 20 correctly classified images and 5 misclassified images, saving them into `gradcam_gallery/`. Plot the UMAP scatter points and save it to `omni_export/umap_scatter.png`. Write metadata linking the images to their UMAP coordinates into `omni_export/metadata.csv`. If you time out, output empty files at these exact paths.

### Stretch Agent (Track Delta)
You are the Stretch Agent. You own `src/stretch.py`. You may not begin until `checkpoints/best_model.pth` exists. You operate in parallel to the Evaluation Agent. Execute two robustness improvements. First, randomly sample 1,000 test images, compress at Q=95, Q=75, Q=50, and Q=25, and plot the accuracy-vs-quality curve showing the degradation cliff, saving it to `results/jpeg_robustness.png`. Second, infer against the WikiArt subset at Q=95, saving confidence scores to `results/wikiart_confidence.json`. If you time out, create a `results/wikiart_confidence.json` recording `{"reason": "WikiArt omitted due to time constraints - refer to README limitations"}`.

### Presentation Agent (Track Epsilon)
You are the Presentation Agent. You own `presentation_outline.md`. You may begin immediately. Maximum 8 slides. Establish a Heist narrative arc. Pair every technical result placeholder with exactly one plain-English sentence. Apply the museum director test to every slide. Output the textual markdown outline into `presentation_outline.md`. If you run out of time, produce outline bullets for the 8 slides.

### Documentation Agent (Track Epsilon)
You are the Documentation Agent. You own `README.md`. You may begin immediately. Write the comprehensive markdown documentation covering VIPER architecture, Kaggle CIFAKE ingestion, Logistic versus EfficientNet results, Omni Data Track elements, and instructions for reproducibility. If Track Delta fails the WikiArt test, you must append a limitations statement at the very end.

### Notebook Agent (Track Epsilon)
You are the Notebook Agent. You own `notebooks/assembly.ipynb`. You are the final aggregation step. You may not begin until Tracks Alpha, Beta, Gamma, and Delta yield their final outputs. Thread all results together into a single master Jupyter notebook. Heavily narrate every EDA figure, baseline metric, and evaluation plot using markdown cells. Do not write training routines here; only assemble and visualize. If you cannot finish entirely, output whatever notebook progress exists.

## Constraint Block
- **Autonomous Track Execution:** Tracks must be executed in parallel wherever boundaries allow. Blocking a parallel track for an unrelated sequential dependency is severely forbidden.
- **Dependency strictness:** Downstream agents within a track will immediately crash if upstream files do not exist at the exact specified paths. Wait for your upstream dependency.
- **Fail-forward execution:** If an agent times out, it must output mock or partial files matching its contract strictly. Failing to produce the file will stall the pipeline. No agent leaves a missing file.
- **Architectural lock-in:** No agent has discretion over architecture, model choice, feature selection, or file naming. Do not change hyperparameters, metrics, or output names. Implement what is written.
- **Path structure:** All outputs must be uniquely named files at named paths. "Results" are not an output; `/results/metrics.json` is an output.

## Verification Checklist
- [ ] **Track Alpha**: `python src/dataloader.py` completes.
- [ ] **Track Beta**: `python src/eda.py` completes and `cat results/feature_matrix.csv` confirms data. `python src/baseline.py` completes.
- [ ] **Track Gamma**: `python src/train.py` yields `checkpoints/best_model.pth`. `python src/evaluate.py` yields core metrics. `python src/visualize.py` yields 25 Grad-CAM maps and Omni maps.
- [ ] **Track Delta**: `python src/stretch.py` yields `results/jpeg_robustness.png` and `results/wikiart_confidence.json`.
- [ ] **Track Epsilon**: `cat presentation_outline.md` verified. `cat README.md` is populated. `jupyter nbconvert --execute notebooks/assembly.ipynb` executes head-to-tail without exceptions.

The plan is complete. Agents may begin Track Alpha and Epsilon.

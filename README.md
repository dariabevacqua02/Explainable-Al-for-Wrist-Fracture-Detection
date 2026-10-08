# Wrist Fracture Detection — Deep Learning Project

> **Advanced Deep Learning** · Data Science MSc  
> Sara Maria Amico · Daria Bevacqua  
> Academic Year 2025/2026

---

## Overview

This project develops an explainable AI system for automatic detection of wrist fractures from X-ray images. A fine-tuned ResNet50 classifier is combined with two complementary explainability techniques, Grad-CAM and Improved Integrated Gradients, and a vision-language model (MedCLIP) used as an independent semantic validator. The pipeline produces human-readable diagnostic reports for each misclassified test image.

The goal is not only to classify, but to localize where the model believes a fracture is present and to cross-validate that localization against the ground-truth YOLO bounding boxes and against MedCLIP's independent image–text similarity signal.

---

## Dataset

**DETECCION DE FRACTURAS** — wrist X-ray dataset in YOLO format [[Roboflow Universe, 2025]](https://universe.roboflow.com/detecciondefracturasantebrazo-8tucq/deteccion-de-fracturas-bntcm).

| Split | Images |
|-------|--------|
| Train | ~2 000 |
| Valid | ~400   |
| Test  | ~400   |

Each image comes with a `.txt` annotation file (YOLO format). Class `0` marks a fracture region with a bounding box; absence of the file (or no class-0 entry) means normal. The dataset is moderately imbalanced toward the normal class, which is handled during training via a dynamically computed `pos_weight`.

---

## System Requirements

- Python 3.8 or later, GPU strongly recommended (Google Colab T4 or A100)
- Core: `torch>=2.0`, `torchvision>=0.15`, `captum>=0.6`, `grad-cam>=1.4`
- Vision-language: `medclip`, `transformers>=4.18`
- Utilities: `scikit-learn`, `Pillow`, `numpy`, `matplotlib`, `pandas`, `opencv-python`

All dependencies are listed in `requirements.txt`.

---

## Usage

### 1. Prepare the dataset

Upload `DETECCION DE FRACTURAS.zip` to Google Drive under `MyDrive/` and mount Drive in Google Colab. The pipeline unzips and indexes the dataset automatically on first run.

### 2. Train the classifier

Fine-tunes a ResNet50 backbone with layer-wise learning rates and early stopping. If a saved checkpoint already exists at `MyDrive/best_model_resnet50_colab.pth`, training is skipped and the checkpoint is loaded directly.

### 3. Evaluate on the test set

Computes accuracy, balanced accuracy, recall, specificity, F1, AUC, confusion matrix, and ROC curve on the held-out test split.

### 4. Generate explainability maps

Produces Grad-CAM heatmaps (on `model.layer4[-1]`) and Improved IG attribution maps (SmoothGrad, blurred baseline) for all four outcome types (TP, TN, FP, FN) and computes localization consistency metrics against the YOLO ground-truth bounding boxes.

### 5. Run MedCLIP validation

Downloads MedCLIP weights automatically from the public GCS bucket on first run. Runs prompt ensembling and computes ResNet–MedCLIP agreement on the test set.

### 6. Inspect diagnostic reports

Structured `.txt` reports are generated for every misclassified test image (FP and FN) and a balanced case study (one TP, FP, FN, TN) is visualized inline with Grad-CAM overlays.

---

## Model

### Architecture

| Component | Detail |
|---|---|
| Base | ResNet50 pretrained on ImageNet |
| Head | `Dropout(0.5)` → `Linear(2048, 1)` |
| Loss | `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` |

### Training setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam with layer-wise LR |
| LR (layer1–2) | 1e-5 |
| LR (layer3–4) | 1e-4 |
| LR (head) | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (patience=3) |
| Early stopping | patience=5 |
| Max epochs | 30 |
| Batch size | 32 |

---

## Explainability

### Grad-CAM

Applies Gradient-weighted Class Activation Mapping on `model.layer4[-1]`, the deepest convolutional block of ResNet50. The resulting heatmap highlights the spatial regions that most influenced the binary prediction.

### Improved Integrated Gradients (SmoothGrad)

Uses Captum's `IntegratedGradients` with a `NoiseTunnel` wrapper (SmoothGrad, 8 samples, σ=0.1) and a Gaussian-blurred baseline (σ=5.0) instead of a black image. A blurred baseline is more appropriate for radiographs than the standard zero-image baseline, which is semantically meaningless for grayscale images with structured background.

### Localization consistency

Both Grad-CAM and Improved IG attribution maps are evaluated against the YOLO ground-truth fracture bounding boxes using two metrics:

- **Pointing-game hit rate** — fraction of fracture images where the peak attribution pixel falls inside the annotated fracture region
- **Energy-in-box** — fraction of total attribution mass that lies inside the annotated region, compared against the chance level (mean box area / image area)

---

## MedCLIP Validation

[MedCLIP ViT](https://github.com/RyanWangZf/MedCLIP) is a CLIP-style model pretrained on chest X-ray image–report pairs, used here as an independent second opinion. Given the same test images, it ranks text prompts by semantic similarity to the image embedding.

Starting from a 5-prompt baseline, five progressively refined prompt sets (v1–v5) were designed with balanced fracture/normal descriptions. The best-performing configuration uses an ensemble of prompts, aggregating fracture and normal scores across multiple descriptions. For each test image, the ResNet prediction is compared with the MedCLIP ensemble prediction; agreement is reported overall and broken down by true class.

---

## Diagnostic Reports

For every misclassified test image (FP and FN), a structured `.txt` report is generated containing:

- Classification result, confidence score, and ground truth
- MedCLIP semantic prediction (fracture vs normal score)
- Most semantically relevant prompt
- ResNet–MedCLIP consistency flag
- Reference to Grad-CAM and IG visual explanations

---

## Results

### ResNet50 classification (test set)

| Metric | Value |
|---|---|
| Accuracy | 0.931 |
| Balanced Accuracy | 0.928 |
| Precision | 0.958 |
| Recall (fracture) | 0.937 |
| Specificity | 0.919 |
| F1 Score | 0.948 |
| AUC | 0.979 |

### Localization consistency vs YOLO ground-truth bounding boxes

**Grad-CAM**

| Metric | Value |
|---|---|
| Pointing-game hit rate | 42.9% |
| Energy-in-box | 12.6% |
| Chance level (mean box area) | 2.6% |
| Lift over chance | 16.5× |

**Improved IG (SmoothGrad)**

| Metric | Value |
|---|---|
| Pointing-game hit rate | 69.6% |
| Energy-in-box | 17.2% |
| Chance level (mean box area) | 2.8% |
| Lift over chance | 25.1× |

### MedCLIP agreement — ensemble prompts

| Metric | Value |
|---|---|
| Overall agreement (ResNet vs MedCLIP) | 63.6% |
| Agreement on Normal images | 59.4% |
| Agreement on Fracture images | 65.8% |

---

## Design Principles

A few choices held throughout the project:

- **Blurred baseline for IG.** A zero-image baseline is semantically meaningless for X-rays. A Gaussian-blurred version of the input image is a more realistic "uncertain" baseline and produces attribution maps that align better with clinical intuition.
- **MedCLIP over generic CLIP.** MedCLIP was trained on medical image–report pairs, so its text encoder understands clinical terminology ("cortical disruption", "distal radius") in a way that OpenAI CLIP — trained on web images and captions — does not.
- **Prompt ensembling over a single prompt.** A single prompt can be ambiguous or poorly calibrated. Averaging scores across multiple semantically related prompts reduces variance and makes the MedCLIP signal more robust.
- **Layer-wise learning rates.** Early feature extractors (low-level edges and textures learned on ImageNet) are kept largely frozen; deeper layers and the classification head are allowed to adapt to the radiological domain.

---

## References

- **Dataset**: [DETECCION DE FRACTURAS — Roboflow Universe, 2025](https://universe.roboflow.com/detecciondefracturasantebrazo-8tucq/deteccion-de-fracturas-bntcm)
- **MedCLIP**: [RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP)

---

## Authors

- **Sara Maria Amico** — saramariaamico@gmail.com  
- **Daria Bevacqua** — bvcdra@gmail.com

Advanced Deep Learning · Data Science MSc · 2025/2026

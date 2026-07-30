# WTB-DefectNet Experiments

Experimental evaluation, ablation studies, performance analysis, and Explainable AI (XAI) visualizations for **WTB-DefectNet**, a deep learning framework for multi-class wind turbine blade defect classification.

---

## Overview

This repository contains the experimental results and evaluation of **WTB-DefectNet**, developed for automatic classification of wind turbine blade surface defects.

The repository documents the complete experimental workflow, including:

- Baseline experiment results
- Training and validation curves
- Performance metrics
- Confusion matrices
- Explainable AI (Grad-CAM) visualizations
- Future ablation studies

---

## Experiment Structure

```
WTB-DefectNet-Experiments/
│
├── Experiment_01_Baseline/
│   ├── Curves/
│   ├── eval/
│   ├── gradcam/
│   ├── log.csv
│   └── README.md
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── plot_curves.py
│
├── requirements.txt
└── README.md
```

---

# Experiment 1 — Baseline

The first experiment establishes the baseline performance of WTB-DefectNet on a 9-class wind turbine blade defect dataset.

### Training Configuration

- Architecture: WTB-DefectNet
- Number of Classes: 9
- Epochs: 120
- Optimizer: AdamW
- Learning Rate Scheduler: Cosine Annealing
- Loss Function: Cross-Entropy Loss

---

# Test Performance

| Metric | Score |
|---------|-------|
| Accuracy | **70.71%** |
| Macro Precision | **63.43%** |
| Macro Recall | **68.72%** |
| Macro F1-Score | **65.37%** |
| Weighted F1-Score | **71.01%** |
| Balanced Accuracy | **68.72%** |
| Cohen's Kappa | **0.6416** |
| Matthews Correlation Coefficient | **0.6435** |

---

# Per-Class Performance

| Defect Class | F1 Score |
|--------------|---------:|
| Oil Leakage | **99.52%** |
| Erosion | **86.56%** |
| Lightning Strikes | **85.07%** |
| Protective Film Damage | **68.91%** |
| Pinholes | **63.25%** |
| Coating Detachment | **61.54%** |
| Paint Cracks | **48.89%** |
| Localized Damage | **46.28%** |
| Surface Stains | **28.35%** |

---

# Key Observations

### Strengths

- Excellent detection of **Oil Leakage**.
- Strong performance on **Erosion** and **Lightning Strikes**.
- Stable convergence during training.
- Good overall classification performance for a challenging multi-class problem.

### Challenges

The model struggles with visually similar defect categories, particularly:

- Surface Stains
- Paint Cracks
- Localized Damage

These defects exhibit overlapping texture and appearance, leading to increased inter-class confusion.

---

# Training Curves

The repository includes:

- Training Loss
- Validation Loss
- Combined Loss Curve
- Validation Macro F1
- Validation Balanced Accuracy

These curves illustrate the learning progression and convergence behaviour throughout training.

---


# Repository Contents

- Experimental logs
- Performance metrics
- Confusion matrices
- Training curves
- Grad-CAM visualizations
- Baseline and future ablation studies

---

## Citation

If you use this work in your research, please cite the corresponding WTB-DefectNet publication once available.

---

## Author

**Mirunalini A**

B.Tech Electronics and Computer Science Engineering

VIT Chennai

GitHub: https://github.com/miruanand

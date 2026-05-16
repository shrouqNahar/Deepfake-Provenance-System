# 🧠 Hybrid DeepFake Detection with Social IoT Provenance



A robust framework for detecting deepfake images by combining **deep learning classification** with **image provenance verification**.  
Designed to withstand the noise and metadata manipulation typical of Social Internet of Things (IoT) environments.

---

##  Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Training & Evaluation](#training--evaluation)
- [Advanced Analyses](#advanced-analyses)
  [Provenance Survivability](#provenance-survivability)
  [Ablation Study (Penalty Sensitivity)](#ablation-study-penalty-sensitivity)
  [Computational Overhead](#computational-overhead)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview
Deepfake detection models often fail when images undergo lossy transformations (resizing, compression, blurring) or when their metadata is tampered with or removed – common in social media and IoT sharing.

This project proposes a hybrid scoring system**:
1. A fine‑tuned EfficientNet‑B0** model provides a base deepfake probability.
2. Provenance metadata** (SHA‑256 hash, perceptual hash) is checked for each image.  
    Missing** metadata → +15% penalty  
    Forged** metadata → +35% penalty  
3. The final hybrid score is clipped to [0, 1] and thresholded at 0.5.

The pipeline also simulates **provenance attacks** on the test set, evaluates survivability of perceptual hashes under Social IoT transformations, and performs a sensitivity analysis of penalty values.

---

## Key Features
-  Deepfake classification** using EfficientNet‑B0 (transfer learning from ImageNet)
-  Provenance generation & verification** with SHA‑256 and perceptual hashing (pHash)
-  Automatic dataset splitting** (train/val/test = 70/15/15) for datasets lacking predefined splits
-  Simulated provenance attacks**: metadata removal (20%) and forgery (10%) on the test set
-  Hybrid scoring** that adjusts model predictions based on metadata integrity
-  Comprehensive analysis**: ROC/PR curves, confusion matrix, survival rates, ablation study
-  Computational overhead** measurement (inference time, parameter count)
-  Modular, clean codebase** – easy to extend and reuse

---

## Pipeline Architecture

Training & Evaluation
Model: EfficientNet-B0 (from timm), pretrained on ImageNet

Input size: 224×224, normalized with ImageNet stats

Batch size: 2 (Colab‑friendly; increase if GPU memory allows)

Optimizer: Adam, lr = 1e‑4

Loss: Cross‑Entropy

Epochs: 6 (configurable in config.py)

Device: Automatically selects GPU if available

After training, the script generates:

classification_report.csv

confusion_matrix.csv (also .png visual)

detailed_results.csv (per‑image scores)

roc_curve_visual.png

score_distribution.png

Advanced Analyses
Provenance Survivability
Simulates typical Social IoT image processing (resizing, JPEG compression, Gaussian blur) with parallel metadata removal/forgery.

Runs multiple times (default 5) to measure how often the perceptual hash remains valid after transformations.

Output: provenance_survivability.csv

Ablation Study (Penalty Sensitivity)
Evaluates the hybrid system across a grid of penalty values:

Missing penalty ∈ {0.05, 0.10, 0.15, 0.20, 0.25}

Forged penalty ∈ {0.25, 0.30, 0.35, 0.40, 0.45}

Compares AUC of detection‑only vs. hybrid for every combination.

Output: ablation_results.csv

Computational Overhead
Measures:

Average inference time per image (ms)

Standard deviation

Number of model parameters (millions)

Printed directly to the console and ready for inclusion in a paper/report.


After running the pipeline you will see:

Hybrid AUC: 0.9823
Average Survivability Rate: 78.4%
Best ablation: miss=0.15, forged=0.35 → AUC gain 0.018
Inference time: 4.32 ms ± 0.23 ms
Params: 4.01 M
All figures and CSV files are saved to the working directory, ready for integration into your LaTeX document or thesis.

Contributing
Contributions are welcome!

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

---

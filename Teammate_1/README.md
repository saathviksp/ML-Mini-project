# Pulsar Candidate Classification — Supervised Learning Pipeline (Teammate 1)

**Course**: UE24CS352A Machine Learning • Mini-Project Assignment  
**Domain**: Astronomical Signal Processing & Binary Classification  
**Reference Paper**: *Application of machine learning methods to identify and categorize radio pulsar signal candidates* (Debesai, Gutierrez, & Koyluoglu, Stanford CS229)  
**Dataset**: HTRU2 Dataset (17,898 candidate signals: 16,259 non-pulsars, 1,639 true pulsars)

---

## 📌 Executive Summary & Work Division (Teammate 1)

This repository contains the complete **Supervised Learning Pipeline** created for **Teammate 1**. Teammate 1 owns the binary classification pipeline (Pulsar vs. Non-Pulsar), implementing:
1. **Exploratory Data Analysis (EDA)**: Class imbalance (1:9.9 ratio), feature probability distributions, and feature correlation matrix.
2. **Gaussian Discriminant Analysis (GDA Baseline from Scratch)**: Custom implementation of GDA derived analytically from first principles ($\phi$, $\mu_0$, $\mu_1$, $\Sigma$).
3. **Scikit-Learn Random Forest**: Hyperparameter optimization using `RandomizedSearchCV` (`n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `criterion`).
4. **Random Forest from Scratch**: Custom Decision Tree & Random Forest Classifier implemented in pure Python/NumPy (Shannon Entropy, Information Gain, Bootstrap Sampling, Random Feature Subsets).
5. **5-Fold Cross-Validation Protocol**: **Fold-Level Upsampling** (upsampling positive class in training folds ONLY; validation and test sets remain untouched raw distributions).
6. **Threshold Optimization Sweep**: Moving decision boundary from $0.5 \rightarrow 0.2$ to maximize pulsar recall (raising recall from ~88% to ~91%-93% as recommended in the benchmark paper).
7. **Project Deliverables**: Executable scripts, interactive Jupyter notebook, publication-grade 2-page PDF report, PowerPoint (.pptx) slide deck, and PDF presentation slides.

---

## 📁 Repository Structure

```
ml_mini_project/
├── data/
│   ├── HTRU_2.csv                 # Raw HTRU2 dataset (17,898 rows x 9 columns)
│   └── Readme.txt                 # Dataset description from UCI ML repository
├── src/
│   ├── __init__.py                # Package initializer
│   ├── eda.py                     # EDA script (class imbalance, distributions, heatmaps)
│   ├── gda.py                     # Custom Gaussian Discriminant Analysis (from scratch)
│   ├── random_forest_scratch.py   # Decision Tree & Random Forest (from scratch in NumPy)
│   ├── random_forest_sklearn.py   # Scikit-Learn RF + RandomizedSearchCV hyperparameter tuner
│   └── evaluation.py              # 5-fold CV with upsampling, ROC/PRC curves, threshold sweep
├── notebooks/
│   └── teammate_1_supervised.ipynb# Interactive Jupyter Demonstration Notebook
├── outputs/
│   ├── plots/                     # High-resolution figures generated for paper & slides
│   │   ├── class_distribution.png # HTRU2 class imbalance bar chart
│   │   ├── feature_distributions.png# KDE distributions by class
│   │   ├── correlation_matrix.png # Feature correlation heatmap
│   │   ├── gda_threshold_sweep.png# GDA Sensitivity / 1-Specificity vs Threshold
│   │   ├── rf_threshold_sweep.png # RF Sensitivity / 1-Specificity vs Threshold
│   │   ├── roc_curve.png          # Comparative ROC curves (AUC)
│   │   └── prc_curve.png          # Comparative Precision-Recall curves (AP)
│   └── metrics_summary.json       # Exported test & 5-fold CV metrics
├── docs/
│   ├── writeup_teammate1.pdf      # 2-Page Project Summary Report (PDF) complying with guidelines
│   ├── slides_teammate1.pptx      # Presentation Slide Deck (PowerPoint .pptx format)
│   └── slides_teammate1.pdf       # Presentation Slide Deck (Landscape PDF format)
├── main_supervised.py             # End-to-end runnable Python pipeline
├── build_notebook.py              # Script to build teammate_1_supervised.ipynb
├── build_writeup_pdf.py           # ReportLab generator for writeup_teammate1.pdf
├── build_slides.py                # Python-pptx generator for slides_teammate1.pptx
├── build_slides_pdf.py            # ReportLab generator for slides_teammate1.pdf
├── requirements.txt               # Dependencies file
└── README.md                      # Complete project setup and execution guide
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Environment Setup
Clone the repository and install required packages:
```bash
# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Execution Instructions

### A. Run End-to-End Supervised Pipeline
To execute the complete pipeline (EDA, GDA baseline, Scikit-Learn RF tuning, Scratch RF, threshold sweeps, ROC/PRC plots, and metrics export):
```bash
python main_supervised.py
```

### B. Launch Interactive Jupyter Notebook
To run and inspect the interactive demonstration notebook:
```bash
jupyter notebook notebooks/teammate_1_supervised.ipynb
```

### C. Regenerate PDF Report & Presentation Slides
To rebuild the 2-page PDF report and presentation slide decks:
```bash
# Rebuild 2-page PDF writeup
python build_writeup_pdf.py

# Rebuild PowerPoint slide deck (.pptx)
python build_slides.py

# Rebuild PDF slide deck (.pdf)
python build_slides_pdf.py
```

---

## 📊 Empirical Results & Model Comparison

Evaluated on the held-out test set (**N=3,580 candidates**, stratified 80/20 train/test split):

| Model Architecture | Decision Threshold | Accuracy | Recall (TPR) | Precision (PPV) | Specificity | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GDA Baseline (Scratch)** | `0.50` | **97.91%** | 89.63% | 87.76% | 98.74% | 0.8869 | **0.9708** |
| **GDA Baseline (Opt. Th.)** | `0.39` | 97.57% | 90.85% | 83.94% | 98.25% | 0.8726 | 0.9708 |
| **Random Forest (Scikit-Learn)** | `0.50` | 97.88% | 88.41% | 88.41% | 98.83% | 0.8841 | 0.9678 |
| **Random Forest (Tuned)** | `0.20` | 96.76% | **91.16%** | 77.46% | 97.32% | 0.8375 | 0.9678 |
| **Random Forest (From Scratch)** | `0.50` | 97.21% | 90.85% | 80.98% | 97.85% | 0.8563 | 0.9752 |
| **RF From Scratch (Th. 0.2)** | `0.20` | 90.95% | **93.29%** | 50.33% | 90.71% | 0.6538 | 0.9752 |

---

## 💡 Threshold Optimization Analysis
In astronomical pulsar discovery, discarding a true pulsar candidate (False Negative) has a significantly higher scientific penalty than inspecting a spurious signal (False Positive). 
- Following the methodology in Section 7.2 of *Debesai et al.*, we performed a threshold sweep across $[0.0, 1.0]$.
- Shifting the decision threshold from **0.5 to ~0.2** raises pulsar recall to **91.16%** (Scikit-Learn RF) and **93.29%** (Scratch RF), while keeping false positives under manageable levels.

---

## 🤝 Handoff for Teammate 2 (Unsupervised Pipeline)

Teammate 2 owns the **Unsupervised Categorization and Interactive Demo App**.
- The 1,639 true pulsar candidate signals (`target == 1`) have been extracted and prepared in `data/HTRU_2.csv`.
- Teammate 2 can import the trained supervised classifier from `src/gda.py` or `src/random_forest_sklearn.py` directly into their Streamlit/Gradio demo app for live candidate predictions.

---

## 📄 License & Course Compliance
Developed for **UE24CS352A Machine Learning Mini-Project**. All code and deliverables strictly comply with assignment instructions and guidelines.

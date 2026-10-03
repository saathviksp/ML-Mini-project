import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Pulsar Candidate Classification — Supervised Learning Pipeline
**Course**: UE24CS352A Machine Learning • Mini-Project  
**Teammate 1**: Supervised Classification (Pulsar vs. Non-Pulsar)  
**Dataset**: HTRU2 (17,898 candidate signals, 8 continuous features, 1,639 pulsars)

---

## 1. Executive Summary & Problem Overview
Astronomical radio surveys produce vast amounts of candidate signals. Most signals are noise or Radio Frequency Interference (RFI). Real pulsar signals are extremely rare (~9.16% of total candidates in HTRU2).

**Teammate 1 Scope**:
1. **Exploratory Data Analysis (EDA)**: Class imbalance, feature probability distributions, correlation matrix.
2. **Gaussian Discriminant Analysis (GDA) Baseline from Scratch**:
   - Class priors $\\phi$, class mean vectors $\\mu_0, \\mu_1$, shared covariance matrix $\\Sigma$.
3. **Scikit-Learn Random Forest (RF)**:
   - Hyperparameter optimization with `RandomizedSearchCV`.
4. **Random Forest from Scratch**:
   - Decision Trees built using Shannon Entropy and Information Gain, Bootstrap sampling, and Random Feature Subsets.
5. **Evaluation Protocol**:
   - Stratified 80/20 train/test split.
   - 5-Fold Cross Validation with **Fold-Level Upsampling** (upsampling training fold ONLY; validation and test folds are untouched).
   - Threshold sweep (moving decision boundary from $0.5 \\rightarrow 0.2$ to maximize recall).
   - Comparative ROC & Precision-Recall (PRC) curves.
"""))

# Cell 2: Imports & Environment Setup
cells.append(nbf.v4.new_code_cell("""import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

from src.eda import load_data, run_eda, FEATURE_NAMES, TARGET_NAME
from src.gda import GaussianDiscriminantAnalysis
from src.random_forest_scratch import RandomForestScratch
from src.random_forest_sklearn import train_and_tune_rf_sklearn
from src.evaluation import (
    upsample_positive_class,
    evaluate_metrics,
    run_5fold_cross_validation,
    plot_threshold_sweep,
    plot_roc_prc_curves
)

%matplotlib inline
"""))

# Cell 3: Data Loading & EDA
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Loading & Exploratory Data Analysis (EDA)"""))

cells.append(nbf.v4.new_code_cell("""# Load HTRU2 dataset
data_path = '../data/HTRU_2.csv' if os.path.exists('../data/HTRU_2.csv') else '../HTRU2_data/HTRU_2.csv'
df = load_data(data_path)
print(f"Dataset shape: {df.shape}")
display(df.head())

# Run EDA and display figures
class_counts = run_eda(df, output_dir='../outputs/plots')

# Display class distribution plot
from IPython.display import Image, display
display(Image('../outputs/plots/class_distribution.png'))
display(Image('../outputs/plots/feature_distributions.png'))
display(Image('../outputs/plots/correlation_matrix.png'))
"""))

# Cell 4: Train/Test Split
cells.append(nbf.v4.new_markdown_cell("""## 3. Stratified Train / Test Split"""))

cells.append(nbf.v4.new_code_cell("""X = df[FEATURE_NAMES].values
y = df[TARGET_NAME].values

# 80/20 Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]} (Non-pulsar: {np.sum(y_train==0)}, Pulsar: {np.sum(y_train==1)})")
print(f"Test samples:     {X_test.shape[0]} (Non-pulsar: {np.sum(y_test==0)}, Pulsar: {np.sum(y_test==1)})")
"""))

# Cell 5: GDA Baseline Implementation & Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 4. Gaussian Discriminant Analysis (GDA Baseline from Scratch)
GDA models $P(x|y)$ as a multivariate Gaussian:
$$y \\sim \\text{Bernoulli}(\\phi)$$
$$x|y=0 \\sim \\mathcal{N}(\\mu_0, \\Sigma)$$
$$x|y=1 \\sim \\mathcal{N}(\\mu_1, \\Sigma)$$

Parameters:
$$\\phi = \\frac{1}{n} \\sum_{i=1}^n 1\\{y^{(i)}=1\\}$$
$$\\mu_0 = \\frac{\\sum_{i=1}^n 1\\{y^{(i)}=0\\} x^{(i)}}{\\sum_{i=1}^n 1\\{y^{(i)}=0\\}}, \\quad \\mu_1 = \\frac{\\sum_{i=1}^n 1\\{y^{(i)}=1\\} x^{(i)}}{\\sum_{i=1}^n 1\\{y^{(i)}=1\\}}$$
$$\\Sigma = \\frac{1}{n} \\sum_{i=1}^n (x^{(i)} - \\mu_{y^{(i)}})(x^{(i)} - \\mu_{y^{(i)}})^T$$
"""))

cells.append(nbf.v4.new_code_cell("""# 5-Fold Cross Validation with fold-level upsampling
gda_cv_results = run_5fold_cross_validation(GaussianDiscriminantAnalysis, X_train, y_train)

# Fit GDA on full upsampled training set
X_train_up, y_train_up = upsample_positive_class(X_train, y_train, random_state=42)
gda = GaussianDiscriminantAnalysis()
gda.fit(X_train_up, y_train_up)

# Evaluate on test set
gda_proba_test = gda.predict_proba(X_test)
gda_test_metrics_05 = evaluate_metrics(y_test, (gda_proba_test >= 0.5).astype(int), gda_proba_test)
print("GDA Test Set Metrics (Threshold = 0.5):")
for k, v in gda_test_metrics_05.items():
    print(f"  {k}: {v:.4f}")

# Threshold sweep plot for GDA
opt_th_gda = plot_threshold_sweep(y_test, gda_proba_test, model_name="GDA Baseline", save_path="../outputs/plots/gda_threshold_sweep.png")
display(Image('../outputs/plots/gda_threshold_sweep.png'))
"""))

# Cell 6: Scikit-Learn Random Forest
cells.append(nbf.v4.new_markdown_cell("""## 5. Random Forest (Scikit-Learn + Hyperparameter Tuning)"""))

cells.append(nbf.v4.new_code_cell("""# Tune hyperparameters using RandomizedSearchCV
best_rf_sk, best_params = train_and_tune_rf_sklearn(X_train_up, y_train_up, random_state=42, n_iter=8, cv=3)

# 5-Fold CV evaluation
rf_sk_cv = run_5fold_cross_validation(best_rf_sk, X_train, y_train, is_sklearn=True)

# Fit tuned model on training set
best_rf_sk.fit(X_train_up, y_train_up)
rf_sk_proba_test = best_rf_sk.predict_proba(X_test)[:, 1]

# Evaluate at 0.5 and 0.2 thresholds
rf_sk_metrics_05 = evaluate_metrics(y_test, (rf_sk_proba_test >= 0.5).astype(int), rf_sk_proba_test)
rf_sk_metrics_02 = evaluate_metrics(y_test, (rf_sk_proba_test >= 0.2).astype(int), rf_sk_proba_test)

print("RF Scikit-Learn (Threshold 0.5):", rf_sk_metrics_05)
print("RF Scikit-Learn (Threshold 0.2):", rf_sk_metrics_02)

# Threshold sweep plot for RF
opt_th_rf = plot_threshold_sweep(y_test, rf_sk_proba_test, model_name="Random Forest (scikit-learn)", save_path="../outputs/plots/rf_threshold_sweep.png")
display(Image('../outputs/plots/rf_threshold_sweep.png'))
"""))

# Cell 7: Random Forest from Scratch
cells.append(nbf.v4.new_markdown_cell("""## 6. Random Forest (Implemented from Scratch)
Built using Decision Trees with:
- Shannon Entropy: $E(S) = - \\sum p(x) \\log_2 p(x)$
- Information Gain: $IG = E(\\text{parent}) - \\frac{N_{\\text{left}}}{N} E(\\text{left}) - \\frac{N_{\\text{right}}}{N} E(\\text{right})$
- Bootstrap Sampling & Random Feature Subsets ($\\\\sqrt{p}$ features)
"""))

cells.append(nbf.v4.new_code_cell("""rf_scratch = RandomForestScratch(n_estimators=10, max_depth=8, min_samples_split=5, random_state=42)
rf_scratch.fit(X_train_up, y_train_up)

rf_scratch_proba_test = rf_scratch.predict_proba(X_test)
rf_scratch_metrics_05 = evaluate_metrics(y_test, (rf_scratch_proba_test >= 0.5).astype(int), rf_scratch_proba_test)
rf_scratch_metrics_02 = evaluate_metrics(y_test, (rf_scratch_proba_test >= 0.2).astype(int), rf_scratch_proba_test)

print("RF Scratch Test Metrics (Threshold 0.5):", rf_scratch_metrics_05)
print("RF Scratch Test Metrics (Threshold 0.2):", rf_scratch_metrics_02)
"""))

# Cell 8: Comparison & Summary Plots
cells.append(nbf.v4.new_markdown_cell("""## 7. Comparative Performance & Curves"""))

cells.append(nbf.v4.new_code_cell("""proba_dict = {
    'GDA Baseline': gda_proba_test,
    'RF (scikit-learn)': rf_sk_proba_test,
    'RF (from scratch)': rf_scratch_proba_test
}
plot_roc_prc_curves(y_test, proba_dict, save_dir='../outputs/plots')

display(Image('../outputs/plots/roc_curve.png'))
display(Image('../outputs/plots/prc_curve.png'))
"""))

nb['cells'] = cells

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/teammate_1_supervised.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook notebooks/teammate_1_supervised.ipynb created successfully.")

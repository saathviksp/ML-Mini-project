import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, mean_squared_error
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import os

def upsample_positive_class(X, y, random_state=42):
    """
    Upsamples the positive class (y == 1) so positive count equals negative count.
    X, y: numpy arrays
    """
    X_pos, y_pos = X[y == 1], y[y == 1]
    X_neg, y_neg = X[y == 0], y[y == 0]

    n_neg = len(y_neg)
    X_pos_upsampled, y_pos_upsampled = resample(
        X_pos, y_pos,
        replace=True,
        n_samples=n_neg,
        random_state=random_state
    )

    X_upsampled = np.vstack((X_neg, X_pos_upsampled))
    y_upsampled = np.hstack((y_neg, y_pos_upsampled))

    # Shuffle
    shuffle_idx = np.random.RandomState(seed=random_state).permutation(len(y_upsampled))
    return X_upsampled[shuffle_idx], y_upsampled[shuffle_idx]

def evaluate_metrics(y_true, y_pred, y_proba=None):
    """Calculates comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    roc_auc_val = auc(*roc_curve(y_true, y_proba)[:2]) if y_proba is not None else None
    ap_val = average_precision_score(y_true, y_proba) if y_proba is not None else None

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'roc_auc': roc_auc_val,
        'pr_auc': ap_val
    }

def run_5fold_cross_validation(model_cls_or_inst, X_train, y_train, is_sklearn=False, **model_kwargs):
    """
    Executes 5-Fold Cross Validation with Fold-level Upsampling.
    Upsamples positive class in training fold ONLY; validation fold is NOT upsampled.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    mse_diffs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # Upsample training fold ONLY
        X_tr_up, y_tr_up = upsample_positive_class(X_tr, y_tr, random_state=42 + fold)

        # Instantiate or clone model
        if is_sklearn:
            from sklearn.base import clone
            model = clone(model_cls_or_inst)
        else:
            model = model_cls_or_inst(**model_kwargs)

        model.fit(X_tr_up, y_tr_up)

        # Train & Val evaluation
        tr_proba = model.predict_proba(X_tr_up)
        if tr_proba.ndim > 1:
            tr_proba = tr_proba[:, 1]
        val_proba = model.predict_proba(X_val)
        if val_proba.ndim > 1:
            val_proba = val_proba[:, 1]

        val_pred = (val_proba >= 0.5).astype(int)
        tr_pred = (tr_proba >= 0.5).astype(int)

        metrics = evaluate_metrics(y_val, val_pred, val_proba)
        
        # MSE difference (val MSE - train MSE)
        tr_mse = mean_squared_error(y_tr_up, tr_proba)
        val_mse = mean_squared_error(y_val, val_proba)
        mse_diff = val_mse - tr_mse
        mse_diffs.append(mse_diff)
        
        fold_results.append(metrics)

    avg_metrics = {k: np.mean([f[k] for f in fold_results if f[k] is not None]) for k in fold_results[0].keys()}
    avg_mse_diff = np.mean(mse_diffs)
    avg_metrics['mean_mse_diff'] = avg_mse_diff

    print("=== 5-Fold CV Results (Upsampled Training Folds Only) ===")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    return avg_metrics

def plot_threshold_sweep(y_true, y_proba, model_name="Model", save_path="outputs/plots/threshold_sweep.png"):
    """
    Plots Sensitivity (TPR), 1-Specificity (FPR), and Difference (TPR - FPR)
    against probability threshold [0.0, 1.0], matching paper Figures 1 and 2.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    tprs = []
    fprs = []
    diffs = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tprs.append(tpr)
        fprs.append(fpr)
        diffs.append(tpr - fpr)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, tprs, 'o-', color='#1f77b4', label='Sensitivity (TPR)', markersize=3)
    plt.plot(thresholds, fprs, 's-', color='#ff7f0e', label='1 - Specificity (FPR)', markersize=3)
    plt.plot(thresholds, diffs, '^-', color='#2ca02c', label='Difference (TPR - FPR)', markersize=3)
    
    # Highlight optimal threshold around 0.2
    opt_idx = np.argmax(diffs)
    opt_th = thresholds[opt_idx]
    plt.axvline(opt_th, color='red', linestyle='--', alpha=0.7, label=f'Optimal Threshold (~{opt_th:.2f})')

    plt.title(f'Sensitivity / 1 - Specificity / Difference vs. Threshold ({model_name})', fontsize=11, fontweight='bold')
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('Sensitivity / 1 - Specificity / Difference', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='center right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return opt_th

def plot_roc_prc_curves(y_true, proba_dict, save_dir="outputs/plots"):
    """Plots comparative ROC and PRC curves for GDA, RF sklearn, RF scratch."""
    os.makedirs(save_dir, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(7, 5))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Baseline')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # PRC Curve
    plt.figure(figsize=(7, 5))
    for name, proba in proba_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.plot(rec, prec, label=f'{name} (AP = {ap:.4f})', linewidth=2)
    plt.title('Precision-Recall (PRC) Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prc_curve.png'), dpi=300)
    plt.close()

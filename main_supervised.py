import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def run_supervised_pipeline():
    print("==================================================================")
    print("   Pulsar Candidate Classification - Supervised Pipeline (Teammate 1)")
    print("==================================================================")
    
    # 1. Load Data
    data_path = 'data/HTRU_2.csv'
    if not os.path.exists(data_path):
        data_path = 'HTRU2_data/HTRU_2.csv'
    df = load_data(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 2. Run EDA
    run_eda(df, output_dir='outputs/plots')
    
    # 3. Train/Test Split (Stratified 80/20)
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples.")
    print(f"Train class balance: {np.bincount(y_train)} | Test class balance: {np.bincount(y_test)}")
    
    results_summary = {}

    # 4. GDA Baseline
    print("\n--- Model 1: Gaussian Discriminant Analysis (GDA Baseline from Scratch) ---")
    gda_cv = run_5fold_cross_validation(GaussianDiscriminantAnalysis, X_train, y_train, is_sklearn=False)
    
    # Fit GDA on full upsampled training set and evaluate on test set
    X_train_up, y_train_up = upsample_positive_class(X_train, y_train, random_state=42)
    gda_model = GaussianDiscriminantAnalysis()
    gda_model.fit(X_train_up, y_train_up)
    
    gda_proba_test = gda_model.predict_proba(X_test)
    gda_test_metrics_05 = evaluate_metrics(y_test, (gda_proba_test >= 0.5).astype(int), gda_proba_test)
    
    # Threshold sweep for GDA
    opt_th_gda = plot_threshold_sweep(y_test, gda_proba_test, model_name="GDA Baseline", save_path="outputs/plots/gda_threshold_sweep.png")
    gda_test_metrics_opt = evaluate_metrics(y_test, (gda_proba_test >= opt_th_gda).astype(int), gda_proba_test)
    
    print("GDA Test Results (Threshold 0.5):", gda_test_metrics_05)
    print(f"GDA Test Results (Optimal Threshold {opt_th_gda:.2f}):", gda_test_metrics_opt)

    results_summary['GDA_Baseline'] = {
        '5fold_cv': gda_cv,
        'test_0.5': gda_test_metrics_05,
        'test_opt': gda_test_metrics_opt
    }

    # 5. Scikit-Learn Random Forest
    print("\n--- Model 2: Random Forest (Scikit-Learn + RandomizedSearchCV) ---")
    best_rf_sklearn, best_params = train_and_tune_rf_sklearn(X_train_up, y_train_up, random_state=42, n_iter=8, cv=3)
    
    rf_sk_cv = run_5fold_cross_validation(best_rf_sklearn, X_train, y_train, is_sklearn=True)
    
    # Fit tuned RF on full upsampled training set
    best_rf_sklearn.fit(X_train_up, y_train_up)
    rf_sk_proba_test = best_rf_sklearn.predict_proba(X_test)[:, 1]
    
    rf_sk_metrics_05 = evaluate_metrics(y_test, (rf_sk_proba_test >= 0.5).astype(int), rf_sk_proba_test)
    
    # Threshold sweep for RF (Targeting threshold ~0.2 as in paper)
    opt_th_rf = plot_threshold_sweep(y_test, rf_sk_proba_test, model_name="Random Forest (scikit-learn)", save_path="outputs/plots/rf_threshold_sweep.png")
    rf_sk_metrics_02 = evaluate_metrics(y_test, (rf_sk_proba_test >= 0.2).astype(int), rf_sk_proba_test)
    rf_sk_metrics_opt = evaluate_metrics(y_test, (rf_sk_proba_test >= opt_th_rf).astype(int), rf_sk_proba_test)

    print("RF Scikit-Learn Test Results (Threshold 0.5):", rf_sk_metrics_05)
    print("RF Scikit-Learn Test Results (Threshold 0.2):", rf_sk_metrics_02)
    print(f"RF Scikit-Learn Test Results (Optimal Threshold {opt_th_rf:.2f}):", rf_sk_metrics_opt)

    results_summary['Random_Forest_Sklearn'] = {
        'best_params': best_params,
        '5fold_cv': rf_sk_cv,
        'test_0.5': rf_sk_metrics_05,
        'test_0.2': rf_sk_metrics_02,
        'test_opt': rf_sk_metrics_opt
    }

    # 6. Random Forest from Scratch
    print("\n--- Model 3: Random Forest (From Scratch) ---")
    rf_scratch = RandomForestScratch(n_estimators=10, max_depth=8, min_samples_split=5, random_state=42)
    rf_scratch.fit(X_train_up, y_train_up)
    
    rf_scratch_proba_test = rf_scratch.predict_proba(X_test)
    rf_scratch_metrics_05 = evaluate_metrics(y_test, (rf_scratch_proba_test >= 0.5).astype(int), rf_scratch_proba_test)
    rf_scratch_metrics_02 = evaluate_metrics(y_test, (rf_scratch_proba_test >= 0.2).astype(int), rf_scratch_proba_test)
    
    print("RF Scratch Test Results (Threshold 0.5):", rf_scratch_metrics_05)
    print("RF Scratch Test Results (Threshold 0.2):", rf_scratch_metrics_02)

    results_summary['Random_Forest_Scratch'] = {
        'test_0.5': rf_scratch_metrics_05,
        'test_0.2': rf_scratch_metrics_02
    }

    # 7. Generate Combined Plots (ROC & PRC)
    proba_dict = {
        'GDA Baseline': gda_proba_test,
        'RF (scikit-learn)': rf_sk_proba_test,
        'RF (from scratch)': rf_scratch_proba_test
    }
    plot_roc_prc_curves(y_test, proba_dict, save_dir='outputs/plots')

    # 8. Save Metrics Summary
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/metrics_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print("\n==================================================================")
    print("   Supervised Pipeline Execution Finished Successfully!")
    print("   Metrics summary saved to outputs/metrics_summary.json")
    print("   Plots saved to outputs/plots/")
    print("==================================================================")

if __name__ == '__main__':
    run_supervised_pipeline()

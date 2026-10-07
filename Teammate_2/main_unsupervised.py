"""End-to-end unsupervised pipeline for pulsar candidate categorization.

Run from inside the Teammate_2 directory:

    python main_unsupervised.py

Produces every plot in outputs/plots, the metrics file
outputs/cluster_metrics.json, and the fitted objects in outputs/models
that the demo app loads.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from src.preprocess import (FEATURE_NAMES, RANDOM_STATE, ensure_dir,
                            load_pulsar_matrix)
from src.kmeans_cluster import (K_BASELINE, cluster_profile, fit_kmeans,
                                plot_k_selection, sweep_k)
from src.som import SelfOrganizingMap
from src.pca_viz import (feature_component_correlation, fit_pca,
                         plot_clusters, plot_explained_variance)
from src.evaluate import (group_som_nodes, plot_silhouette,
                          plot_som_diagnostics, plot_som_node_groups,
                          silhouette_report, som_sample_labels)

DATA_PATH = os.path.join('data', 'HTRU_2.csv')
PLOT_DIR = os.path.join('outputs', 'plots')
MODEL_DIR = os.path.join('outputs', 'models')
METRICS_PATH = os.path.join('outputs', 'cluster_metrics.json')

SOM_ROWS = 5
SOM_COLS = 5
SOM_EPOCHS = 100
SOM_LEARNING_RATE = 0.5


def load_supervised_model():
    """Trains Teammate 1's from-scratch GDA for use in the demo app.

    The demo has to answer two questions about a candidate: is it a pulsar,
    and if so which group does it resemble. The first is a supervised
    question, so rather than duplicate that work the GDA implementation
    from the supervised half is imported and fitted on the full dataset.
    """
    t1_src = os.path.abspath(os.path.join('..', 'Teammate_1', 'src'))
    if t1_src not in sys.path:
        sys.path.insert(0, t1_src)
    from gda import GaussianDiscriminantAnalysis

    df = pd.read_csv(os.path.join('..', 'Teammate_1', 'data', 'HTRU_2.csv'),
                     header=None)
    df.columns = FEATURE_NAMES + ['target']
    X = df[FEATURE_NAMES].values
    y = df['target'].values

    model = GaussianDiscriminantAnalysis().fit(X, y)
    return model


def main():
    ensure_dir(PLOT_DIR)
    ensure_dir(MODEL_DIR)

    print('=== Loading HTRU2 and isolating the pulsar class ===')
    X_raw, X_scaled, scaler = load_pulsar_matrix(DATA_PATH)
    print(f'Confirmed pulsars retained: {len(X_raw)}')
    print(f'Features: {len(FEATURE_NAMES)}')

    # ------------------------------------------------------------------
    # Choosing k
    # ------------------------------------------------------------------
    print('\n=== Sweeping k for k-means (elbow and silhouette) ===')
    ks, inertias, silhouettes = sweep_k(X_scaled)
    for k, inertia, sil in zip(ks, inertias, silhouettes):
        print(f'k={k:2d}  inertia={inertia:9.1f}  silhouette={sil:.4f}')
    plot_k_selection(ks, inertias, silhouettes, PLOT_DIR, K_BASELINE)

    # ------------------------------------------------------------------
    # K-means baseline
    # ------------------------------------------------------------------
    print(f'\n=== K-means baseline (k={K_BASELINE}) ===')
    kmeans_model, kmeans_labels = fit_kmeans(X_scaled, K_BASELINE)
    profile = cluster_profile(X_raw, kmeans_labels)
    print(profile.round(3).to_string())

    # ------------------------------------------------------------------
    # Self-Organizing Map
    # ------------------------------------------------------------------
    print(f'\n=== Training {SOM_ROWS}x{SOM_COLS} SOM from scratch ===')
    som = SelfOrganizingMap(
        rows=SOM_ROWS,
        cols=SOM_COLS,
        input_dim=X_scaled.shape[1],
        learning_rate=SOM_LEARNING_RATE,
        n_epochs=SOM_EPOCHS,
        random_state=RANDOM_STATE,
    )
    som.fit(X_scaled)
    print(f'Quantization error after epoch 1:   '
          f'{som.quantization_errors_[0]:.4f}')
    print(f'Quantization error after epoch {SOM_EPOCHS}: '
          f'{som.quantization_errors_[-1]:.4f}')

    occupied = int(np.count_nonzero(som.hit_map(X_scaled)))
    print(f'Occupied nodes: {occupied} of {som.n_nodes}')

    node_groups = group_som_nodes(som, K_BASELINE)
    som_labels = som_sample_labels(som, X_scaled, node_groups)
    som_bmu_labels = som_sample_labels(som, X_scaled)

    plot_som_diagnostics(som, X_scaled, PLOT_DIR)
    plot_som_node_groups(som, node_groups, PLOT_DIR)

    # ------------------------------------------------------------------
    # PCA, for visualisation only
    # ------------------------------------------------------------------
    print('\n=== PCA projection for visualisation ===')
    pca_full, _ = fit_pca(X_scaled, n_components=X_scaled.shape[1])
    plot_explained_variance(pca_full, PLOT_DIR)
    ratios = pca_full.explained_variance_ratio_
    print(f'PC1 explains {ratios[0] * 100:.2f}% of variance')
    print(f'PC2 explains {ratios[1] * 100:.2f}% of variance')
    print(f'PC1+PC2 together: {(ratios[0] + ratios[1]) * 100:.2f}%')

    pca, X_pca = fit_pca(X_scaled, n_components=2)
    plot_clusters(
        X_pca,
        {f'K-means (k={K_BASELINE})': kmeans_labels,
         f'SOM {SOM_ROWS}x{SOM_COLS} grouped into {K_BASELINE}': som_labels},
        pca,
        PLOT_DIR,
    )
    corr_df, _ = feature_component_correlation(X_scaled, X_pca, PLOT_DIR)
    print('\nFeature to component correlation:')
    print(corr_df.round(3).to_string())

    # ------------------------------------------------------------------
    # Silhouette comparison
    # ------------------------------------------------------------------
    print('\n=== Silhouette scores ===')
    scores = silhouette_report(X_scaled, {
        f'kmeans_k{K_BASELINE}': kmeans_labels,
        f'som_grouped_k{K_BASELINE}': som_labels,
        'som_raw_bmu': som_bmu_labels,
    })
    for name, value in scores.items():
        print(f'{name:22s} {value:.4f}')

    plot_silhouette(X_scaled, kmeans_labels,
                    f'Silhouette, k-means (k={K_BASELINE})',
                    PLOT_DIR, 'silhouette_kmeans.png')
    plot_silhouette(X_scaled, som_labels,
                    f'Silhouette, SOM grouped into {K_BASELINE}',
                    PLOT_DIR, 'silhouette_som.png')

    # ------------------------------------------------------------------
    # Persist everything the demo app needs
    # ------------------------------------------------------------------
    print('\n=== Saving fitted objects and metrics ===')
    supervised = load_supervised_model()

    artefacts = {
        'scaler': scaler,
        'kmeans': kmeans_model,
        'som': som,
        'som_node_groups': node_groups,
        'pca': pca,
        'gda': supervised,
        'cluster_profile': profile,
    }
    with open(os.path.join(MODEL_DIR, 'unsupervised_artifacts.pkl'), 'wb') as f:
        pickle.dump(artefacts, f)

    metrics = {
        'dataset': {
            'total_candidates': 17898,
            'pulsars_used': int(len(X_raw)),
            'n_features': len(FEATURE_NAMES),
        },
        'k_selection': {
            'k_values': ks,
            'inertia': [float(v) for v in inertias],
            'silhouette': [float(v) for v in silhouettes],
            'k_chosen': K_BASELINE,
        },
        'kmeans': {
            'k': K_BASELINE,
            'silhouette': scores[f'kmeans_k{K_BASELINE}'],
            'inertia': float(kmeans_model.inertia_),
            'cluster_sizes': {str(int(c)): int(n) for c, n in
                              zip(profile.index, profile['n_candidates'])},
        },
        'som': {
            'grid': [SOM_ROWS, SOM_COLS],
            'epochs': SOM_EPOCHS,
            'initial_learning_rate': SOM_LEARNING_RATE,
            'initial_sigma': float(som.sigma0),
            'quantization_error_first_epoch':
                float(som.quantization_errors_[0]),
            'quantization_error_final_epoch':
                float(som.quantization_errors_[-1]),
            'occupied_nodes': occupied,
            'silhouette_grouped': scores[f'som_grouped_k{K_BASELINE}'],
            'silhouette_raw_bmu': scores['som_raw_bmu'],
        },
        'pca': {
            'explained_variance_ratio': [float(v) for v in ratios],
            'pc1_pc2_cumulative': float(ratios[0] + ratios[1]),
        },
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Metrics written to {METRICS_PATH}')
    print(f'Plots written to {PLOT_DIR}')
    print('Pipeline complete.')


if __name__ == '__main__':
    main()

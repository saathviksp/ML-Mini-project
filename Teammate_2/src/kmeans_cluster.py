import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .preprocess import RANDOM_STATE, ensure_dir

K_BASELINE = 3
K_RANGE = range(2, 11)


def fit_kmeans(X_scaled, k=K_BASELINE, random_state=RANDOM_STATE):
    """Fits k-means on the scaled pulsar matrix.

    n_init=10 restarts Lloyd's algorithm from ten different seedings and
    keeps the run with the lowest inertia, which guards against the
    algorithm settling in a poor local optimum.
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    return model, labels


def sweep_k(X_scaled, k_range=K_RANGE, random_state=RANDOM_STATE):
    """Runs k-means for each k and records inertia and silhouette score.

    Inertia (within-cluster sum of squares) always falls as k grows, so it
    is read as an elbow rather than a maximum. The silhouette score does
    have a meaningful maximum, so the two are used together.
    """
    inertias = []
    silhouettes = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, silhouettes


def plot_k_selection(ks, inertias, silhouettes, output_dir='outputs/plots',
                     chosen_k=K_BASELINE):
    """Saves the elbow curve and silhouette curve side by side."""
    ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ks, inertias, marker='o', color='#1f77b4')
    axes[0].axvline(chosen_k, linestyle='--', color='grey', linewidth=1)
    axes[0].set_title('Elbow method (within-cluster SSE)', fontweight='bold')
    axes[0].set_xlabel('Number of clusters k')
    axes[0].set_ylabel('Inertia')
    axes[0].set_xticks(ks)

    axes[1].plot(ks, silhouettes, marker='o', color='#ff7f0e')
    axes[1].axvline(chosen_k, linestyle='--', color='grey', linewidth=1)
    axes[1].set_title('Silhouette score by k', fontweight='bold')
    axes[1].set_xlabel('Number of clusters k')
    axes[1].set_ylabel('Mean silhouette')
    axes[1].set_xticks(ks)

    plt.tight_layout()
    path = os.path.join(output_dir, 'kmeans_k_selection.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def cluster_profile(X_raw, labels):
    """Mean of every raw (unscaled) feature within each cluster.

    Reported in the write-up so the clusters can be described in physical
    terms rather than as anonymous group numbers.
    """
    profile = X_raw.copy()
    profile['cluster'] = labels
    summary = profile.groupby('cluster').mean()
    summary['n_candidates'] = profile.groupby('cluster').size()
    return summary

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .preprocess import FEATURE_NAMES, RANDOM_STATE, ensure_dir


def fit_pca(X_scaled, n_components=2, random_state=RANDOM_STATE):
    """Fits PCA on the scaled pulsar matrix.

    PCA is used here for visualisation only. Clustering is performed in
    the full 8-dimensional space, and the components are fitted
    afterwards purely to get a 2D view of the result. Clustering on the
    projection instead would throw away whatever variance the discarded
    components carry.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def plot_explained_variance(pca_full, output_dir='outputs/plots'):
    """Scree plot plus the cumulative explained variance curve."""
    ensure_dir(output_dir)
    ratios = pca_full.explained_variance_ratio_
    comps = np.arange(1, len(ratios) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(comps, ratios, color='#1f77b4')
    axes[0].set_title('Variance explained per component', fontweight='bold')
    axes[0].set_xlabel('Principal component')
    axes[0].set_ylabel('Proportion of variance')
    axes[0].set_xticks(comps)

    axes[1].plot(comps, np.cumsum(ratios), marker='o', color='#ff7f0e')
    axes[1].axhline(0.9, linestyle='--', color='grey', linewidth=1)
    axes[1].set_title('Cumulative variance explained', fontweight='bold')
    axes[1].set_xlabel('Number of components')
    axes[1].set_ylabel('Cumulative proportion')
    axes[1].set_xticks(comps)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pca_explained_variance.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_clusters(X_pca, label_sets, pca, output_dir='outputs/plots',
                  filename='cluster_scatter_pca.png'):
    """Scatters the candidates on PC1/PC2, one panel per clustering method.

    `label_sets` maps a panel title to its array of cluster labels.
    """
    ensure_dir(output_dir)
    n_panels = len(label_sets)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.6),
                             squeeze=False)
    var = pca.explained_variance_ratio_

    for ax, (title, labels) in zip(axes[0], label_sets.items()):
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=14,
                             cmap='viridis', alpha=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var[0] * 100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({var[1] * 100:.1f}% variance)')
        legend = ax.legend(*scatter.legend_elements(), title='Cluster',
                           loc='best', fontsize=8)
        ax.add_artist(legend)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def feature_component_correlation(X_scaled, X_pca, output_dir='outputs/plots'):
    """Correlates each original feature with PC1 and PC2.

    This is what makes the projection interpretable: a component is only
    meaningful once you know which physical measurements load onto it.
    """
    ensure_dir(output_dir)
    corr = np.zeros((len(FEATURE_NAMES), X_pca.shape[1]))
    for i in range(len(FEATURE_NAMES)):
        for j in range(X_pca.shape[1]):
            corr[i, j] = np.corrcoef(X_scaled[:, i], X_pca[:, j])[0, 1]

    corr_df = pd.DataFrame(
        corr,
        index=FEATURE_NAMES,
        columns=[f'PC{j + 1}' for j in range(X_pca.shape[1])]
    )

    plt.figure(figsize=(5, 5))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson r'})
    plt.title('Feature to component correlation', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'pca_feature_correlation.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return corr_df, path

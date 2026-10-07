import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from .preprocess import RANDOM_STATE, ensure_dir


def group_som_nodes(som, k=3, random_state=RANDOM_STATE):
    """Groups the SOM codebook vectors into k clusters.

    A 5x5 SOM produces 25 micro-clusters, which is not directly comparable
    with a k-means run at k=3. The standard fix (Vesanto and Alhoniemi,
    2000) is a two-level approach: cluster the 25 codebook vectors rather
    than the 1,639 original samples, then map each sample to the group of
    its BMU. This keeps the comparison like for like and is far cheaper
    than clustering the raw data again.

    Returns the array mapping node index to group id.
    """
    grouper = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    node_groups = grouper.fit_predict(som.weights)
    return node_groups


def som_sample_labels(som, X_scaled, node_groups=None):
    """Cluster label per sample.

    With `node_groups` supplied the label is the group of the sample's
    BMU; without it the label is the raw BMU index.
    """
    bmu = som.predict(X_scaled)
    if node_groups is None:
        return bmu
    return node_groups[bmu]


def silhouette_report(X_scaled, label_sets):
    """Mean silhouette score for each named clustering."""
    scores = {}
    for name, labels in label_sets.items():
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            scores[name] = None
            continue
        scores[name] = float(silhouette_score(X_scaled, labels))
    return scores


def plot_silhouette(X_scaled, labels, title, output_dir='outputs/plots',
                    filename='silhouette_kmeans.png'):
    """Per-sample silhouette plot for one clustering.

    Each cluster appears as a horizontal band. Samples with values below
    the dashed mean line sit close to a neighbouring cluster, and negative
    values are likely misassigned, so the shape of the bands says more
    than the single average does.
    """
    ensure_dir(output_dir)
    values = silhouette_samples(X_scaled, labels)
    mean_score = float(np.mean(values))
    unique_labels = np.unique(labels)

    plt.figure(figsize=(6.5, 4.8))
    y_lower = 10
    cmap = plt.cm.viridis
    for i, label in enumerate(unique_labels):
        cluster_values = np.sort(values[labels == label])
        size = len(cluster_values)
        y_upper = y_lower + size
        colour = cmap(i / max(len(unique_labels) - 1, 1))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                          facecolor=colour, edgecolor=colour, alpha=0.8)
        plt.text(-0.05, y_lower + 0.5 * size, str(label))
        y_lower = y_upper + 10

    plt.axvline(mean_score, color='red', linestyle='--', linewidth=1,
                label=f'Mean = {mean_score:.3f}')
    plt.title(title, fontweight='bold')
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Candidates grouped by cluster')
    plt.yticks([])
    plt.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    return mean_score, path


def plot_som_diagnostics(som, X_scaled, output_dir='outputs/plots'):
    """U-matrix, hit map and the quantization error curve in one figure."""
    ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    sns.heatmap(som.u_matrix(), ax=axes[0], cmap='bone_r', annot=True,
                fmt='.2f', cbar_kws={'label': 'Mean distance to neighbours'})
    axes[0].set_title('U-matrix (cluster boundaries)', fontweight='bold')

    sns.heatmap(som.hit_map(X_scaled), ax=axes[1], cmap='YlGnBu', annot=True,
                fmt='d', cbar_kws={'label': 'Candidates mapped'})
    axes[1].set_title('Hit map (node occupancy)', fontweight='bold')

    axes[2].plot(np.arange(1, len(som.quantization_errors_) + 1),
                 som.quantization_errors_, color='#2ca02c')
    axes[2].set_title('Quantization error per epoch', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean distance to BMU')

    plt.tight_layout()
    path = os.path.join(output_dir, 'som_diagnostics.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_som_node_groups(som, node_groups, output_dir='outputs/plots'):
    """Shows which grid nodes fall into which of the k final groups."""
    ensure_dir(output_dir)
    grid = node_groups.reshape(som.rows, som.cols)

    plt.figure(figsize=(5, 4.2))
    sns.heatmap(grid, annot=True, fmt='d', cmap='viridis', cbar=False,
                linewidths=0.5, linecolor='white')
    plt.title('SOM nodes grouped into final clusters', fontweight='bold')
    plt.xlabel('Grid column')
    plt.ylabel('Grid row')
    plt.tight_layout()
    path = os.path.join(output_dir, 'som_node_groups.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

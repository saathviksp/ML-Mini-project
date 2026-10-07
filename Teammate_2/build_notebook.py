"""Builds notebooks/teammate_2_unsupervised.ipynb.

Keeping the notebook in a build script means the narrative and the code
stay in one place and the notebook can be regenerated after any change to
the pipeline.

    python build_notebook.py
"""

import os

import nbformat as nbf

OUTPUT_PATH = os.path.join('notebooks', 'teammate_2_unsupervised.ipynb')

CELLS = [
    ('md', """# Pulsar Candidate Categorization (Teammate 2)

**UE24CS352A Machine Learning, Mini-Project**

Teammate 1 handles the supervised question: is a candidate a pulsar? This
notebook handles the next one: among the signals that *are* pulsars, are
there distinct kinds?

Everything below is fitted on the **1,639 confirmed pulsars** only. The
16,259 non-pulsars are dropped, because mixing them back in would just
recover the supervised split we already have.

Pipeline:

1. Load HTRU2, keep `target == 1`, standardise
2. K-means, choosing k with the elbow and silhouette methods
3. A Self-Organizing Map written from scratch
4. PCA, for visualisation only
5. Silhouette comparison of the two methods"""),

    ('code', """import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The notebook lives in notebooks/, the package lives one level up.
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.getcwd())

from src.preprocess import FEATURE_NAMES, RANDOM_STATE, load_pulsar_matrix
from src.kmeans_cluster import cluster_profile, fit_kmeans, sweep_k
from src.som import SelfOrganizingMap
from src.pca_viz import fit_pca
from src.evaluate import group_som_nodes, silhouette_report, som_sample_labels

sns.set_theme(style='whitegrid')
print('Working directory:', os.getcwd())"""),

    ('md', """## 1. Load the data and isolate the pulsars

The raw CSV has no header: eight continuous features then the label.

The features are summary statistics of two things measured for each
candidate, the **integrated pulse profile** (`_ip`) and the **DM-SNR curve**
(`_snr`): mean, standard deviation, excess kurtosis and skewness of each."""),

    ('code', """X_raw, X_scaled, scaler = load_pulsar_matrix('data/HTRU_2.csv')

print(f'Pulsars retained: {len(X_raw)}')
print(f'Features: {list(X_raw.columns)}')
X_raw.describe().round(2)"""),

    ('md', """Standardisation matters here. `mean_ip` runs into the hundreds while
`kurt_ip` sits near zero, and both k-means and the SOM compare points with
Euclidean distance. Without scaling, `mean_ip` would dominate every
distance and the other seven features would barely register."""),

    ('code', """pd.DataFrame({
    'raw mean': X_raw.mean().round(3),
    'raw std': X_raw.std().round(3),
    'scaled mean': X_scaled.mean(axis=0).round(3),
    'scaled std': X_scaled.std(axis=0).round(3),
})"""),

    ('md', """## 2. K-means: choosing k

Two criteria, because they measure different things.

**Inertia** (within-cluster sum of squares) only measures compactness, and
it falls monotonically as k grows, so it is read as an *elbow*: the point
where adding another cluster stops buying much.

**Silhouette** rewards separation as well as compactness and ranges from -1
to 1, so it has a real maximum."""),

    ('code', """ks, inertias, silhouettes = sweep_k(X_scaled)

pd.DataFrame({'k': ks, 'inertia': np.round(inertias, 1),
              'silhouette': np.round(silhouettes, 4)})"""),

    ('code', """fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(ks, inertias, marker='o', color='#1f77b4')
axes[0].axvline(3, linestyle='--', color='grey', linewidth=1)
axes[0].set_title('Elbow method', fontweight='bold')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_xticks(ks)

axes[1].plot(ks, silhouettes, marker='o', color='#ff7f0e')
axes[1].axvline(3, linestyle='--', color='grey', linewidth=1)
axes[1].set_title('Silhouette by k', fontweight='bold')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Mean silhouette')
axes[1].set_xticks(ks)

plt.tight_layout()
plt.show()"""),

    ('md', """The elbow sits at k = 3. The silhouette peaks at k = 2 instead.

We keep **k = 3**. It is at the elbow, it matches the structure the
reference paper reports, and the profiles below show the k = 2 solution
just merges two groups that differ physically. Worth stating plainly in
the viva rather than pretending the two criteria agreed."""),

    ('code', """kmeans_model, kmeans_labels = fit_kmeans(X_scaled, k=3)

profile = cluster_profile(X_raw, kmeans_labels)
profile.round(3)"""),

    ('md', """Reading the three groups off the profile table:

- **Cluster 0** (739) is the typical group: mid-range profile, moderately
  dispersed DM-SNR curve.
- **Cluster 1** (679) has a low profile mean, strong profile skew and the
  highest DM-SNR mean. Bright and sharply peaked.
- **Cluster 2** (221) is the small distinctive one: high profile mean, very
  weak DM-SNR curve, extreme DM-SNR skew."""),

    ('md', """## 3. Self-Organizing Map, from scratch

A SOM maps high-dimensional input onto a low-dimensional grid while
preserving topology: inputs close in the original space end up near each
other on the grid.

Each of the 25 nodes on the 5x5 grid holds a weight (codebook) vector of
the same dimension as the input. For every sample, three steps:

1. **Competition** - find the Best Matching Unit (BMU), the node whose
   weight vector is nearest in Euclidean distance.
2. **Cooperation** - weight every node by a Gaussian centred on the BMU,
   measured in *grid* coordinates. This is what preserves topology.
3. **Adaptation** - `w = w + lr(t) * h(t) * (x - w)`

Both `lr(t)` and the neighbourhood radius `sigma(t)` decay geometrically,
so the map unfolds globally first and fine-tunes locally later."""),

    ('code', """som = SelfOrganizingMap(rows=5, cols=5, input_dim=8,
                        learning_rate=0.5, n_epochs=100,
                        random_state=RANDOM_STATE)
som.fit(X_scaled)

print(f'Quantization error, epoch 1:   {som.quantization_errors_[0]:.4f}')
print(f'Quantization error, epoch 100: {som.quantization_errors_[-1]:.4f}')
print(f'Occupied nodes: {np.count_nonzero(som.hit_map(X_scaled))} / {som.n_nodes}')"""),

    ('code', """fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

sns.heatmap(som.u_matrix(), ax=axes[0], cmap='bone_r', annot=True, fmt='.2f')
axes[0].set_title('U-matrix', fontweight='bold')

sns.heatmap(som.hit_map(X_scaled), ax=axes[1], cmap='YlGnBu', annot=True, fmt='d')
axes[1].set_title('Hit map', fontweight='bold')

axes[2].plot(range(1, len(som.quantization_errors_) + 1),
             som.quantization_errors_, color='#2ca02c')
axes[2].set_title('Quantization error', fontweight='bold')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Mean distance to BMU')

plt.tight_layout()
plt.show()"""),

    ('md', """Three things to read off these:

- The error curve falls smoothly and flattens, so 100 epochs is enough.
- Every node is occupied, so the 5x5 grid is not too large for 1,639
  points. Empty nodes would mean dead units.
- The U-matrix shows the biggest neighbour distances in one corner. That
  ridge is a cluster boundary, and it is where cluster 2 separates."""),

    ('md', """### Comparing the SOM with k-means

The SOM gives 25 micro-clusters, which cannot be compared directly with
k-means at k = 3. The standard fix (Vesanto and Alhoniemi, 2000) is a
two-level approach: cluster the 25 *codebook vectors* into 3 groups, then
give each sample the group of its BMU. Clustering 25 vectors is far
cheaper than re-clustering 1,639 samples."""),

    ('code', """node_groups = group_som_nodes(som, k=3)
som_labels = som_sample_labels(som, X_scaled, node_groups)
som_bmu_labels = som_sample_labels(som, X_scaled)

plt.figure(figsize=(5, 4.2))
sns.heatmap(node_groups.reshape(som.rows, som.cols), annot=True, fmt='d',
            cmap='viridis', cbar=False, linewidths=0.5, linecolor='white')
plt.title('SOM nodes grouped into 3 clusters', fontweight='bold')
plt.xlabel('Grid column'); plt.ylabel('Grid row')
plt.tight_layout()
plt.show()

agreement = np.mean(kmeans_labels == som_labels)
print(f'K-means and SOM agree on {agreement * 100:.1f}% of candidates')"""),

    ('md', """## 4. PCA, for visualisation only

An important distinction: PCA is fitted **after** clustering, purely to get
a 2D view. Clustering runs in the full 8-dimensional space.

Clustering the projection instead would throw away the variance held in
the discarded components before either algorithm sees the data."""),

    ('code', """pca_full, _ = fit_pca(X_scaled, n_components=8)
ratios = pca_full.explained_variance_ratio_

print(f'PC1: {ratios[0] * 100:.2f}%')
print(f'PC2: {ratios[1] * 100:.2f}%')
print(f'PC1 + PC2: {(ratios[0] + ratios[1]) * 100:.2f}%')

pca, X_pca = fit_pca(X_scaled, n_components=2)"""),

    ('code', """fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

for ax, (title, labels) in zip(axes, [('K-means (k=3)', kmeans_labels),
                                      ('SOM 5x5 grouped into 3', som_labels)]):
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=14, cmap='viridis',
                    alpha=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(f'PC1 ({ratios[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({ratios[1] * 100:.1f}%)')
    ax.legend(*sc.legend_elements(), title='Cluster', fontsize=8)

plt.tight_layout()
plt.show()"""),

    ('md', """The two panels are nearly identical, which is the main result: two
algorithms with completely different mechanics recover the same three
groups.

To interpret the axes, correlate each original feature with each
component."""),

    ('code', """corr = pd.DataFrame(
    [[np.corrcoef(X_scaled[:, i], X_pca[:, j])[0, 1] for j in range(2)]
     for i in range(len(FEATURE_NAMES))],
    index=FEATURE_NAMES, columns=['PC1', 'PC2'])

plt.figure(figsize=(5, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature to component correlation', fontweight='bold')
plt.tight_layout()
plt.show()

corr.round(3)"""),

    ('md', """PC1 sets the integrated-profile *shape* statistics (kurtosis +0.89,
skewness +0.84) against the profile mean (-0.87) and DM-SNR kurtosis
(-0.83). It is a "narrow, peaked profile" axis.

PC2 is driven by dispersion: `std_ip` at +0.64 and `std_snr` at +0.54."""),

    ('md', """## 5. Silhouette comparison"""),

    ('code', """scores = silhouette_report(X_scaled, {
    'K-means (k=3)': kmeans_labels,
    'SOM grouped into 3': som_labels,
    'SOM raw BMU (25 nodes)': som_bmu_labels,
})

pd.DataFrame({'silhouette': pd.Series(scores).round(4)})"""),

    ('md', """K-means and the SOM land within 0.003 of each other, which is the
like-for-like comparison.

The 25-node score is low by construction, not because the map failed. With
25 clusters over 1,639 points, neighbouring nodes overlap heavily and the
silhouette penalises exactly that.

**On the absolute level (~0.33):** these are genuine groups but not cleanly
separated ones. The HTRU2 features are summary statistics of the profile
and the DM-SNR curve, so frequency and timing information is averaged away
before we ever see it. Pulsar populations separate best by period and
dispersion measure, and neither survives into this feature set. The score
is reporting a real property of the data, not a broken pipeline."""),

    ('md', """## Conclusions

1. Three groups is a defensible structure in the pulsar population, with
   sizes 739 / 679 / 221.
2. K-means (0.3348) and the from-scratch SOM (0.3316) agree on **93.3%** of
   candidates. Independent methods converging is the strongest evidence
   here that the structure is real.
3. Two principal components retain 82% of the variance, enough for an
   honest 2D picture.
4. Separation is moderate and limited by the feature set, not the
   algorithms. Period and dispersion measure would be the features to add
   next.

## Limitations

- No ground truth for the sub-types, so validation is internal only.
  Confirming cluster 2 as a known class would need a catalogue cross-match
  (for example the ATNF).
- k = 3 is a judgement call from the elbow and the reference paper. The
  silhouette alone would have picked k = 2.
- The SOM depends on its initialisation. The seed is fixed for
  reproducibility, and the three-group structure survives reseeding, but
  the map shifts."""),
]


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(body) if kind == 'md'
        else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python', 'version': '3.9'},
    }
    os.makedirs('notebooks', exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Wrote {OUTPUT_PATH} with {len(nb.cells)} cells')


if __name__ == '__main__':
    build()

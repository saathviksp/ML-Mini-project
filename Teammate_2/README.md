# Unsupervised Categorization and Demo (Teammate 2)

**Course**: UE24CS352A Machine Learning, Mini-Project
**Dataset**: HTRU2 (17,898 candidates, 8 features, 1,639 confirmed pulsars)
**Reference**: Debesai, Gutierrez and Koyluoglu, *Application of machine learning methods to identify and categorize radio pulsar signal candidates*, Stanford CS229

Teammate 1 answers "is this candidate a pulsar". This half answers the next
question: **given that it is a pulsar, what kind of pulsar is it?** All
clustering here is fitted on the 1,639 confirmed pulsars only, with the
16,259 non-pulsars dropped.

---

## What is implemented

| Component | Notes |
| :--- | :--- |
| Preprocessing | Filter to `target == 1`, standardise to zero mean and unit variance |
| K-means baseline | scikit-learn, k = 3, with an elbow and silhouette sweep over k = 2 to 10 |
| Self-Organizing Map | Written from scratch in NumPy, 5x5 grid, BMU search, Gaussian neighbourhood, geometric decay of learning rate and radius |
| PCA | Visualisation only, never used as clustering input |
| Evaluation | Silhouette scores for both methods, U-matrix, hit map, quantization error curve |
| Demo | Streamlit app giving a pulsar prediction plus a cluster assignment |

---

## Setup

```bash
cd Teammate_2
pip install -r requirements.txt
```

Python 3.9 or newer.

## Running the pipeline

```bash
python main_unsupervised.py
```

Takes about 20 seconds. Writes:

- `outputs/plots/` - eight figures used in the write-up and slides
- `outputs/cluster_metrics.json` - every number quoted below
- `outputs/models/unsupervised_artifacts.pkl` - fitted scaler, k-means, SOM, PCA and GDA, loaded by the demo

## Running the demo

```bash
streamlit run app/demo_app.py
```

Opens on <http://localhost:8501>. Enter the eight measurements in the sidebar,
or start from a preset (median pulsar, median non-pulsar, or a random row), and
the app returns:

1. a pulsar or non-pulsar verdict with its probability, from Teammate 1's GDA,
2. the k-means cluster and the SOM cluster, plus the Best Matching Unit
   coordinates on the 5x5 grid,
3. a PCA scatter plot marking where the candidate falls among known pulsars.

The pipeline must be run at least once first, since the app loads the saved
artifacts.

## Notebook

```bash
jupyter notebook notebooks/teammate_2_unsupervised.ipynb
```

Walks through the same pipeline cell by cell with the plots inline.

---

## Repository layout

```
Teammate_2/
├── data/
│   └── HTRU_2.csv                       # same file Teammate 1 uses
├── src/
│   ├── preprocess.py                    # loading, pulsar filtering, scaling
│   ├── kmeans_cluster.py                # k-means, elbow and silhouette sweep
│   ├── som.py                           # Self-Organizing Map from scratch
│   ├── pca_viz.py                       # PCA projection and loadings
│   └── evaluate.py                      # silhouette and SOM diagnostics
├── notebooks/
│   └── teammate_2_unsupervised.ipynb
├── app/
│   └── demo_app.py                      # Streamlit demo
├── outputs/
│   ├── plots/                           # generated figures
│   ├── models/                          # pickled fitted objects
│   └── cluster_metrics.json
├── docs/
│   ├── writeup_teammate2.pdf
│   ├── slides_teammate2.pptx
│   └── slides_teammate2.pdf
├── main_unsupervised.py
├── requirements.txt
└── README.md
```

---

## Results

### Choosing k

| k | Inertia | Silhouette |
| :-: | ---: | ---: |
| 2 | 7942.5 | **0.3676** |
| 3 | 5988.9 | 0.3348 |
| 4 | 4815.1 | 0.3425 |
| 5 | 4091.3 | 0.3063 |
| 6 | 3593.7 | 0.2857 |

Inertia falls fastest up to k = 3 and then flattens, which is the elbow. The
silhouette score disagrees slightly and peaks at k = 2. We kept **k = 3**: it
sits at the elbow, it matches the three-group structure the reference paper
reports, and the k = 2 solution simply merges two groups that the cluster
profiles show are physically distinct. This disagreement is reported rather
than hidden, since the two criteria measure different things. Inertia measures
compactness only, while the silhouette also rewards separation, and on data
this continuous the fewer-cluster solution tends to win on separation.

### Cluster profiles (mean of each raw feature)

| Cluster | n | mean_ip | std_ip | kurt_ip | skew_ip | mean_snr | std_snr | kurt_snr | skew_snr |
| :-: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 | 739 | 71.57 | 42.39 | 2.04 | 6.82 | 38.65 | 57.91 | 2.48 | 8.04 |
| 1 | 679 | 28.62 | 33.19 | 5.01 | 29.19 | 77.09 | 66.12 | 1.06 | 1.52 |
| 2 | 221 | 93.18 | 43.37 | 1.01 | 2.87 | 3.44 | 21.99 | 8.90 | 101.43 |

- **Cluster 0** is the largest and most typical group: a mid-range integrated
  profile with a moderately dispersed DM-SNR curve.
- **Cluster 1** has a low integrated-profile mean, strong profile skew and the
  highest DM-SNR mean. These are bright, sharply peaked candidates.
- **Cluster 2** is the smallest and most distinctive: a high integrated-profile
  mean, a very weak DM-SNR curve and extreme DM-SNR skew.

### Method comparison

| Method | Clusters | Silhouette |
| :--- | :-: | ---: |
| K-means | 3 | 0.3348 |
| SOM (5x5, grouped into 3) | 3 | 0.3316 |
| SOM (raw BMU assignment) | 25 | 0.1715 |

The SOM is trained as 25 nodes, so to compare it with k-means at k = 3 the 25
codebook vectors are clustered into three groups and each candidate inherits
the group of its BMU. This two-level approach follows Vesanto and Alhoniemi
(2000) and is much cheaper than re-clustering all 1,639 samples.

Two independent methods land within 0.003 silhouette of each other and agree on
the group for **93.3%** of candidates, which is good evidence the three-group
structure is real and not an artefact of either algorithm.

The raw 25-node score is low by construction, not because the map is bad: 25
clusters over 1,639 points makes neighbouring nodes overlap heavily, and the
silhouette penalises that.

### SOM training

- Quantization error falls from **2.028** after the first epoch to **0.861**
  after 100 epochs, converging smoothly.
- All **25 of 25** nodes are occupied, so there are no dead units and the grid
  is not over-sized for the data.
- The U-matrix shows its largest inter-node distances in one corner, which is
  the boundary separating the small, distinctive cluster 2 from the rest.

### PCA

PC1 explains **58.8%** of the variance and PC2 a further **23.2%**, so two
components retain **82.0%**. PC1 contrasts the integrated-profile shape
statistics (kurtosis loads at +0.89, skewness at +0.84) against the profile
mean (-0.87) and the DM-SNR kurtosis (-0.83). It is effectively a
"narrow, peaked profile" axis. PC2 is dominated by DM-SNR dispersion
(std_snr at +0.54, std_ip at +0.64).

PCA is used for plotting only. Clustering runs in the full 8-dimensional space,
because projecting first would discard the 18% of variance held in the
remaining components before the algorithms ever see it.

---

## Limitations

1. **Silhouette scores are modest (around 0.33).** The HTRU2 features are
   summary statistics of the integrated profile and the DM-SNR curve, so
   frequency and timing information is already averaged away. Pulsar
   populations are better separated by period and dispersion measure, neither
   of which survives into this feature set. The groups found here are real but
   not cleanly separated, which is what the scores report.
2. **No ground-truth labels for the sub-types**, so the clusters can only be
   validated internally. Confirming that cluster 2 corresponds to a known
   pulsar class would need a cross-match against a catalogue such as the ATNF.
3. **k = 3 is a judgement call**, taken from the elbow and the reference paper.
   The silhouette alone would have chosen k = 2.
4. **The SOM is sensitive to initialisation.** Weights are seeded from random
   training rows and the run is fixed with `random_state=42` for
   reproducibility, but a different seed shifts the map slightly, though the
   three-group structure persists.

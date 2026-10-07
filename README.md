# Pulsar Candidate Classification — ML Mini-Project

**Course**: UE24CS352A Machine Learning • Mini-Project Assignment  
**Team**: Team of 2  
**Dataset**: HTRU2 (17,898 candidates, 8 continuous features, 1,639 pulsars)  
**Reference Paper**: *Application of machine learning methods to identify and categorize radio pulsar signal candidates* (Debesai, Gutierrez, & Koyluoglu, Stanford CS229)

---

## 📁 Repository Structure & Team Division

```
ML-Mini-project/
├── Teammate_1/                  # Supervised Classification Pipeline (Pulsar vs. Non-Pulsar)
│   ├── data/
│   │   └── HTRU_2.csv           # HTRU2 Dataset
│   ├── src/                     # Python modules (EDA, GDA from scratch, RF sklearn, RF scratch, evaluation)
│   ├── notebooks/
│   │   └── teammate_1_supervised.ipynb  # Interactive Demonstration Notebook
│   ├── outputs/
│   │   ├── plots/               # High-res diagnostic plots (class distribution, correlation, ROC/PRC, sweeps)
│   │   └── metrics_summary.json # Summary of test and 5-fold CV metrics
│   ├── docs/
│   │   ├── writeup_teammate1.pdf# 2-Page Project Summary Writeup (PDF)
│   │   ├── slides_teammate1.pptx# Presentation Slides (PowerPoint .pptx)
│   │   └── slides_teammate1.pdf # Presentation Slides (PDF)
│   ├── main_supervised.py       # Main end-to-end runnable script for Teammate 1
│   ├── requirements.txt         # Teammate 1 Python dependencies
│   └── README.md                # Detailed guide & results summary for Teammate 1
│
├── Teammate_2/                  # Unsupervised Categorization (what kind of pulsar?)
│   ├── data/
│   │   └── HTRU_2.csv           # HTRU2 Dataset
│   ├── src/                     # Python modules (preprocessing, k-means, SOM from scratch, PCA, evaluation)
│   ├── notebooks/
│   │   └── teammate_2_unsupervised.ipynb  # Interactive Demonstration Notebook
│   ├── app/
│   │   └── demo_app.py          # Streamlit demo (prediction + cluster assignment)
│   ├── outputs/
│   │   ├── plots/               # Cluster scatter, k selection, SOM diagnostics, silhouette
│   │   ├── models/              # Pickled scaler, k-means, SOM, PCA and GDA
│   │   └── cluster_metrics.json # Silhouette scores, k sweep, SOM and PCA metrics
│   ├── docs/
│   │   ├── writeup_teammate2.pdf# 2-Page Project Summary Writeup (PDF)
│   │   ├── slides_teammate2.pptx# Presentation Slides (PowerPoint .pptx)
│   │   └── slides_teammate2.pdf # Presentation Slides (PDF)
│   ├── main_unsupervised.py     # Main end-to-end runnable script for Teammate 2
│   ├── requirements.txt         # Teammate 2 Python dependencies
│   └── README.md                # Detailed guide & results summary for Teammate 2
│
├── Debesai_Gutierrez_Koyluoglu.pdf                    # Reference Paper
├── Guidelines and Instructions_Mini Project Assignment (1).pdf # Project Guidelines
└── Work_Division_Pulsar_Mini_Project (2).pdf          # Team Work Division Matrix
```

---

## 🚀 How to Run Teammate 1's Supervised Pipeline

Navigate into the `Teammate_1` directory:
```bash
cd Teammate_1

# Install requirements
pip install -r requirements.txt

# Run main supervised classification pipeline
python main_supervised.py

# Launch interactive notebook
jupyter notebook notebooks/teammate_1_supervised.ipynb
```

For detailed mathematical formulations, 5-fold cross-validation upsampling logic, threshold tuning analysis, and full empirical comparison tables, please refer to [`Teammate_1/README.md`](Teammate_1/README.md).

---

## 🔭 How to Run Teammate 2's Unsupervised Pipeline

Where Teammate 1 asks "is this candidate a pulsar", Teammate 2 asks "given that it is, what kind of pulsar is it". All clustering is fitted on the 1,639 confirmed pulsars only.

Navigate into the `Teammate_2` directory:
```bash
cd Teammate_2

# Install requirements
pip install -r requirements.txt

# Run the full unsupervised pipeline (k-means, SOM, PCA, silhouette)
python main_unsupervised.py

# Launch the interactive demo app
streamlit run app/demo_app.py

# Launch interactive notebook
jupyter notebook notebooks/teammate_2_unsupervised.ipynb
```

`main_unsupervised.py` takes about 20 seconds and must be run once before the demo app, which loads the fitted models it saves.

**Headline results:** three groups of 739, 679 and 221 pulsars. K-means (silhouette 0.3348) and the from-scratch SOM (0.3316) agree on 93.3% of candidates. Two principal components retain 82.0% of the variance.

For the SOM mechanics, the k selection argument, cluster profiles and limitations, see [`Teammate_2/README.md`](Teammate_2/README.md).

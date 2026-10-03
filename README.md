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

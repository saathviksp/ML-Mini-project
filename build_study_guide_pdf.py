import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and add total page count in footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 756, "UE24CS352A ML Mini-Project • Teammate 1 Comprehensive Study & Code Guide")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 45, 576, 45)
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 32, page_str)
        self.drawString(36, 32, "Pulsar Candidate Classification — Supervised Learning Pipeline")
        self.restoreState()

def generate_study_guide_pdf(pdf_path='docs/Teammate1_Study_Guide.pdf'):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor('#1A365D')
    BLUE = colors.HexColor('#2B6CB0')
    TEAL = colors.HexColor('#0D9488')
    DARK = colors.HexColor('#2D3748')
    LIGHT_BG = colors.HexColor('#F7FAFC')
    BORDER_COLOR = colors.HexColor('#E2E8F0')
    ACCENT = colors.HexColor('#DD6B20')

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=NAVY, alignment=1, spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=15,
        textColor=BLUE, alignment=1, spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'H1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=16,
        textColor=NAVY, spaceBefore=12, spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'H2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=13,
        textColor=BLUE, spaceBefore=8, spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=12.5,
        textColor=DARK, spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=DARK, spaceAfter=3, leftIndent=12
    )

    code_style = ParagraphStyle(
        'CodeSnippet', parent=styles['Normal'],
        fontName='Courier', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#1E293B'), spaceBefore=3, spaceAfter=4
    )

    qa_question_style = ParagraphStyle(
        'QAQ', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.5, leading=13,
        textColor=ACCENT, spaceBefore=8, spaceAfter=2
    )

    qa_answer_style = ParagraphStyle(
        'QAA', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=DARK, spaceAfter=6, leftIndent=8
    )

    story = []

    # Title & Banner
    story.append(Paragraph("UE24CS352A Machine Learning Mini-Project", title_style))
    story.append(Paragraph("<b>Teammate 1 Complete Study & Code Explanation Guide</b><br/>Supervised Pulsar Candidate Classification (GDA, Random Forest, 5-Fold CV & Threshold Tuning)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=10))

    # SECTION 1: PROJECT OVERVIEW
    story.append(Paragraph("1. High-Level Project Overview & Problem Statement", h1_style))
    p1 = (
        "<b>Astronomical Problem:</b> Radio pulsar search surveys (such as the High Time Resolution Universe Survey, HTRUS) "
        "collect millions of signal candidates. However, the vast majority are spurious noise or Radio Frequency Interference (RFI) "
        "from human technology (e.g. FM radio, radar). Scientists need machine learning to filter candidates automatically.<br/><br/>"
        "<b>HTRU2 Dataset Breakdown:</b><br/>"
        "• Total Candidate Signals: <b>17,898</b><br/>"
        "• Spurious Noise / RFI (Class 0): <b>16,259 (90.84%)</b><br/>"
        "• True Pulsars (Class 1): <b>1,639 (9.16%)</b><br/>"
        "• Imbalance Ratio: ~1:9.9 (For every 1 true pulsar, there are nearly 10 non-pulsars).<br/>"
        "• <b>Features (8 continuous variables):</b><br/>"
        "&nbsp;&nbsp;1-4: Mean, Standard Deviation, Excess Kurtosis, and Skewness of the <i>Integrated Pulse Profile</i>.<br/>"
        "&nbsp;&nbsp;5-8: Mean, Standard Deviation, Excess Kurtosis, and Skewness of the <i>DM-SNR Curve</i>."
    )
    story.append(Paragraph(p1, body_style))

    # SECTION 2: MATHEMATICS & ALGORITHMS
    story.append(Paragraph("2. Mathematical Formulations & Algorithms Explained", h1_style))
    
    # 2.1 GDA
    story.append(Paragraph("2.1 Gaussian Discriminant Analysis (GDA Baseline from Scratch)", h2_style))
    gda_math = (
        "GDA is a <b>generative classifier</b> that models the probability distribution of features $x$ given class $y$, i.e. $P(x|y)$. "
        "It assumes that $P(x|y=0)$ and $P(x|y=1)$ follow multivariate Gaussian distributions with individual mean vectors $\\mu_0$ and $\\mu_1$, "
        "but share a common covariance matrix $\\Sigma$.<br/><br/>"
        "<b>1. Class Prior Probability (Bernoulli distribution):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$$\\phi = P(y=1) = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(y^{(i)} = 1)$$<br/>"
        "<b>2. Class Mean Vectors:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$$\\mu_0 = \\frac{\\sum_{i=1}^N \\mathbb{I}(y^{(i)}=0) x^{(i)}}{\\sum_{i=1}^N \\mathbb{I}(y^{(i)}=0)}, \\quad \\mu_1 = \\frac{\\sum_{i=1}^N \\mathbb{I}(y^{(i)}=1) x^{(i)}}{\\sum_{i=1}^N \\mathbb{I}(y^{(i)}=1)}$$<br/>"
        "<b>3. Shared Covariance Matrix $\\Sigma$:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$$\\Sigma = \\frac{1}{N} \\sum_{i=1}^N (x^{(i)} - \\mu_{y^{(i)}})(x^{(i)} - \\mu_{y^{(i)}})^T$$<br/>"
        "<b>4. Classification Probability $P(y=1|x)$ (Bayes Rule):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;By applying Bayes' Rule, the posterior log-likelihood ratio simplifies to a linear decision boundary:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$$P(y=1|x) = \\frac{1}{1 + \\exp(-(Q_1(x) - Q_0(x)))}$$<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;where $Q_k(x) = -\\frac{1}{2}(x - \\mu_k)^T \\Sigma^{-1}(x - \\mu_k) + \\log P(y=k)$."
    )
    story.append(Paragraph(gda_math, body_style))

    # 2.2 Random Forest
    story.append(Paragraph("2.2 Random Forest & Decision Trees (From Scratch)", h2_style))
    rf_math = (
        "Random Forest is an <b>ensemble algorithm</b> that combines multiple decision trees to reduce variance (overfitting) via "
        "<b>Bagging (Bootstrap Aggregation)</b> and <b>Random Feature Subsets</b>.<br/><br/>"
        "<b>1. Shannon Entropy $E(S)$:</b> Measures impurity in a dataset subset $S$:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$$E(S) = - p_0 \\log_2(p_0) - p_1 \\log_2(p_1)$$<br/>"
        "<b>2. Information Gain ($IG$):</b> Evaluates the quality of a feature split at a node:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$$IG = E(\\text{parent}) - \\left( \\frac{N_{\\text{left}}}{N} E(\\text{left}) + \\frac{N_{\\text{right}}}{N} E(\\text{right}) \\right)$$<br/>"
        "<b>3. Random Feature Subsets:</b> At every node split, only a random subset of $m = \\lfloor\\sqrt{p}\\rfloor = \\lfloor\\sqrt{8}\\rfloor = 2$ features is considered.<br/>"
        "<b>4. Bootstrap Aggregation:</b> Each tree is trained on a random sample of size $N$ drawn with replacement from the training set."
    )
    story.append(Paragraph(rf_math, body_style))

    # SECTION 3: CODE MODULE WALKTHROUGH
    story.append(Paragraph("3. Code Module Architecture & Line-by-Line Breakdown", h1_style))
    
    code_explanations = [
        ("src/eda.py", "Handles dataset loading, class imbalance calculation, KDE probability density plots, and feature correlation matrix heatmap."),
        ("src/gda.py", "Custom GaussianDiscriminantAnalysis class. `fit(X, y)` computes phi, mu0, mu1, and sigma using NumPy matrix operations (`X.T @ X / n`). `predict_proba(X)` calculates Mahalanobis distances and sigmoid probability."),
        ("src/random_forest_scratch.py", "DecisionTreeScratch & RandomForestScratch classes. Calculates Shannon entropy and Information Gain for continuous feature thresholds, uses recursive node splitting, and averages tree predictions."),
        ("src/random_forest_sklearn.py", "Wraps `RandomForestClassifier` with `RandomizedSearchCV` over n_estimators, max_depth, max_features, min_samples_split, and criterion, optimizing for Recall."),
        ("src/evaluation.py", "Includes `upsample_positive_class()` for random oversampling, `run_5fold_cross_validation()` with strict fold-level upsampling, `plot_threshold_sweep()` to shift boundary from 0.5 to 0.2, and ROC/PRC plotting functions."),
        ("main_supervised.py", "Main execution entry point coordinating EDA, 80/20 train/test split, GDA baseline, Scikit-Learn RF tuning, Scratch RF, threshold sweeps, and JSON metrics export.")
    ]
    
    for filename, desc in code_explanations:
        story.append(Paragraph(f"<b>📄 {filename}</b>", h2_style))
        story.append(Paragraph(desc, body_style))

    # SECTION 4: 5-FOLD CV & UPSAMPLING PROTOCOL
    story.append(Paragraph("4. 5-Fold Cross-Validation & Data Leakage Prevention", h1_style))
    cv_text = (
        "<b>Why Upsample?</b> Due to the 1:9.9 class imbalance, un-sampled models tend to predict the majority class (non-pulsars) "
        "most of the time, leading to poor recall on pulsar candidates.<br/><br/>"
        "<b>The Data Leakage Trap (CRITICAL):</b> If you upsample the entire dataset <i>before</i> splitting into train/test or cross-validation folds, "
        "identical synthetic copies of the positive examples will appear in both the training set and the validation/test set. This causes artificially inflated 'fake' 99%+ accuracy.<br/><br/>"
        "<b>Our Correct Protocol:</b><br/>"
        "1. Split data into 80% Train and 20% Held-out Test.<br/>"
        "2. Divide the training set into 5 folds using `StratifiedKFold`.<br/>"
        "3. In each fold, upsample the positive class in the <b>training fold ONLY</b> (to ~50% positive).<br/>"
        "4. Evaluate the model on the <b>validation fold WITHOUT upsampling</b> (raw 1:9.9 distribution).<br/>"
        "5. Finally, evaluate on the untouched test set."
    )
    story.append(Paragraph(cv_text, body_style))

    # SECTION 5: THRESHOLD TUNING ANALYSIS
    story.append(Paragraph("5. Threshold Sweep Optimization (0.5 → 0.2 Boundary Shift)", h1_style))
    thresh_text = (
        "In standard binary classification, a decision threshold of 0.5 is used. However, in astronomical signal search:<br/>"
        "• <b>False Negative (FN):</b> Discarding a true pulsar star candidate. (<i>EXTREMELY HIGH SCIENTIFIC COST</i> - lost discovery).<br/>"
        "• <b>False Positive (FP):</b> Flagging noise as a pulsar candidate. (<i>LOW COST</i> - astronomers will briefly inspect it).<br/><br/>"
        "Following Section 7.2 of the CS229 paper (Debesai et al.), we performed a threshold sweep from 0.0 to 1.0. "
        "Lowering the decision threshold to <b>0.2</b> raised recall from <b>88.41% → 91.16%</b> (Scikit-Learn RF) and <b>90.85% → 93.29%</b> (Scratch RF)."
    )
    story.append(Paragraph(thresh_text, body_style))

    # SECTION 6: VIVA & Q&A PREPARATION
    story.append(Paragraph("6. viva & Presentation Q&A Preparation (Questions & Answers)", h1_style))
    
    qa_list = [
        ("Q1: Why did you implement GDA as a baseline instead of Naive Bayes or Logistic Regression?",
         "GDA is a generative model designed for continuous features. Naive Bayes assumes features are independent given class, whereas GDA models feature dependencies using a full covariance matrix Σ. Logistic Regression is discriminative; GDA makes stronger multivariate normal assumptions which are asymptotically efficient when feature distributions are near-normal."),
        
        ("Q2: How did you ensure your cross-validation results did not suffer from data leakage?",
         "We used fold-level upsampling. Inside each of the 5 cross-validation folds, Random Minority Oversampling was applied to the training fold ONLY. The validation fold was kept raw and untouched. This guaranteed zero synthetic data leakage into validation or testing."),
        
        ("Q3: What formulas did you use for your GDA implementation from scratch?",
         "We calculated prior φ = N1/N, class means μ0 = Σ x_i(y=0)/N0 and μ1 = Σ x_i(y=1)/N1, and shared covariance matrix Σ = (1/N) Σ (x_i - μ_yi)(x_i - μ_yi)^T. Posterior P(y=1|x) was computed using the sigmoid of the log likelihood ratio: 1 / (1 + exp(-(Q1 - Q0)))."),
        
        ("Q4: How does your Random Forest from scratch split nodes?",
         "At each node, it calculates Shannon Entropy E(S) = -p0 log2(p0) - p1 log2(p1) and Information Gain IG = E(parent) - (N_left/N E_left + N_right/N E_right). It randomly selects √p features and evaluates percentiles to find the feature and threshold that maximize Information Gain."),
        
        ("Q5: Why did you lower the decision threshold from 0.5 to 0.2?",
         "In pulsar search, discarding a true pulsar (False Negative) is far worse than inspecting a noise candidate (False Positive). Lowering the decision threshold to 0.2 maximizes Recall (Sensitivity), raising pulsar candidate detection to 91.16%-93.29%."),
        
        ("Q6: What hyperparameters did RandomizedSearchCV select for Random Forest?",
         "It selected n_estimators=200, min_samples_split=2, max_features='log2', max_depth=110, and criterion='entropy', optimizing directly for Recall score."),
        
        ("Q7: How does Teammate 1's output connect to Teammate 2's work?",
         "Teammate 1 filters raw candidate signals and isolates true pulsar instances (1,639 positive candidates). Teammate 2 takes these positive pulsar candidates to run unsupervised clustering (K-means, Self-Organizing Map, PCA visualization) and demo app.")
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", qa_question_style))
        story.append(Paragraph(a, qa_answer_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Study Guide PDF generated successfully at {pdf_path}")

if __name__ == '__main__':
    generate_study_guide_pdf()

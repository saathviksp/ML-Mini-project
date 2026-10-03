import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_writeup_pdf(pdf_path='docs/writeup_teammate1.pdf'):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1A365D'),
        alignment=1,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#4A5568'),
        alignment=1,
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=6,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4
    )

    formula_style = ParagraphStyle(
        'Formula',
        parent=styles['Normal'],
        fontName='Courier-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0D9488'),
        alignment=1,
        spaceBefore=2,
        spaceAfter=4
    )

    story = []
    
    # Title Block
    story.append(Paragraph("UE24CS352A Machine Learning • Mini-Project Report", title_style))
    story.append(Paragraph("<b>Supervised Pulsar Candidate Classification (Teammate 1 Half)</b><br/>Dataset: HTRU2 | Reference: Debesai, Gutierrez & Koyluoglu (CS229)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E0'), spaceAfter=6))
    
    # 1. Problem Statement & Dataset
    story.append(Paragraph("1. Problem Statement & HTRU2 Dataset", h1_style))
    p1_text = (
        "Radio pulsar search surveys produce millions of candidate signals, the vast majority of which are non-pulsar "
        "noise and Radio Frequency Interference (RFI). Automatic identification of true pulsars is critical for gravitational wave "
        "and cosmic research. The HTRU2 dataset comprises <b>17,898 total candidates</b>, featuring <b>16,259 non-pulsars (90.84%)</b> "
        "and <b>1,639 true pulsars (9.16%)</b>. Each example is defined by 8 continuous variables derived from the integrated pulse profile "
        "and DM-SNR curve (Mean, Standard Deviation, Excess Kurtosis, and Skewness for both)."
    )
    story.append(Paragraph(p1_text, body_style))
    
    # 2. Mathematical Formulation & Supervised Models
    story.append(Paragraph("2. Supervised Learning Methods & Mathematical Formulations", h1_style))
    gda_text = (
        "<b>Gaussian Discriminant Analysis (GDA Baseline from Scratch):</b> GDA models class-conditional distributions as multivariate Gaussians:<br/>"
        "&nbsp;&nbsp;y ~ Bernoulli(&phi;), &nbsp; x|y=0 ~ N(&mu;<sub>0</sub>, &Sigma;), &nbsp; x|y=1 ~ N(&mu;<sub>1</sub>, &Sigma;)<br/>"
        "Parameters &phi;, &mu;<sub>0</sub>, &mu;<sub>1</sub>, and shared covariance matrix &Sigma; were derived analytically from scratch:"
    )
    story.append(Paragraph(gda_text, body_style))
    story.append(Paragraph("&phi; = (1/N)&Sigma; 1{y<sup>(i)</sup>=1}, &nbsp;&nbsp; &mu;<sub>k</sub> = &Sigma; 1{y<sup>(i)</sup>=k}x<sup>(i)</sup> / &Sigma; 1{y<sup>(i)</sup>=k}, &nbsp;&nbsp; &Sigma; = (1/N)&Sigma; (x<sup>(i)</sup> - &mu;<sub>y<sup>(i)</sup></sub>)(x<sup>(i)</sup> - &mu;<sub>y<sup>(i)</sup></sub>)<sup>T</sup>", formula_style))
    
    rf_text = (
        "<b>Random Forest (Scikit-Learn & From Scratch):</b> Random Forest reduces model variance through bootstrap aggregation (bagging) and random feature sub-selection. "
        "Decision tree splits are selected by maximizing Information Gain based on Shannon Entropy:<br/>"
        "&nbsp;&nbsp;E(S) = -&Sigma; p(x) log<sub>2</sub> p(x), &nbsp;&nbsp;&nbsp; IG = E(parent) - (N<sub>left</sub>/N) E(left) - (N<sub>right</sub>/N) E(right)"
    )
    story.append(Paragraph(rf_text, body_style))

    # 3. Experimental Methodology & Upsampling Protocol
    story.append(Paragraph("3. Experimental Methodology & 5-Fold CV Protocol", h1_style))
    exp_text = (
        "To address severe class imbalance (1:9.9 ratio), we established an <b>80/20 stratified train/test split</b>. "
        "To prevent data leakage during hyperparameter selection, <b>Random Minority Oversampling (ROS)</b> was applied to the "
        "<b>training folds ONLY</b> within a 5-Fold Cross-Validation pipeline. Validation folds and held-out test sets remained strictly un-sampled."
    )
    story.append(Paragraph(exp_text, body_style))

    # Table of Results
    story.append(Paragraph("<b>Table 1: Supervised Model Performance Summary (Held-out Test Set, N=3,580)</b>", body_style))
    
    table_data = [
        ['Model Architecture', 'Decision Th.', 'Accuracy', 'Recall (TPR)', 'Precision', 'Specificity', 'F1-Score', 'ROC AUC'],
        ['GDA Baseline (Scratch)', '0.50', '97.91%', '89.63%', '87.76%', '98.74%', '0.8869', '0.9708'],
        ['GDA Baseline (Opt. Th.)', '0.39', '97.57%', '90.85%', '83.94%', '98.25%', '0.8726', '0.9708'],
        ['Random Forest (Sklearn)', '0.50', '97.88%', '88.41%', '88.41%', '98.83%', '0.8841', '0.9678'],
        ['Random Forest (Tuned 0.2)', '0.20', '96.76%', '91.16%', '77.46%', '97.32%', '0.8375', '0.9678'],
        ['RF From Scratch', '0.50', '97.21%', '90.85%', '80.98%', '97.85%', '0.8563', '0.9752'],
        ['RF From Scratch (Th. 0.2)', '0.20', '90.95%', '93.29%', '50.33%', '90.71%', '0.6538', '0.9752']
    ]
    
    t = Table(table_data, colWidths=[110, 55, 48, 55, 48, 52, 45, 48])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # Figures side by side or vertical
    story.append(Paragraph("4. Threshold Optimization & Diagnostic Curves", h1_style))
    thresh_text = (
        "In pulsar discovery, false negatives (discarding a true pulsar star) carry significantly higher scientific cost than false positives. "
        "As conducted in the benchmark paper (Debesai et al.), we performed a <b>threshold sweep [0.0, 1.0]</b> to evaluate Sensitivity, 1-Specificity, and TPR-FPR difference. "
        "Shifting the decision boundary from 0.5 to 0.2 maximized recall (achieving <b>91.16% to 93.29% recall</b>), matching paper findings."
    )
    story.append(Paragraph(thresh_text, body_style))

    # Add images if exist
    img_rf_path = 'outputs/plots/rf_threshold_sweep.png'
    img_roc_path = 'outputs/plots/roc_curve.png'
    
    if os.path.exists(img_rf_path) and os.path.exists(img_roc_path):
        img_table = Table([
            [RLImage(img_rf_path, width=240, height=155), RLImage(img_roc_path, width=240, height=155)]
        ], colWidths=[260, 260])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table)

    story.append(Paragraph("5. Key Findings & Conclusions", h1_style))
    conc_text = (
        "• <b>Baseline vs. Target Model:</b> GDA provided a robust linear baseline with 97.91% accuracy and 89.63% recall. "
        "Random Forest tuned with RandomizedSearchCV achieved outstanding ROC-AUC (0.9678) and strong recall stability.<br/>"
        "• <b>Scratch Implementation Integrity:</b> Both custom GDA and custom Random Forest closely matched scikit-learn models.<br/>"
        "• <b>Teammate 2 Handoff:</b> Positive predictions and true pulsar candidates (N=1,639) are formatted for Teammate 2's unsupervised pipeline (PCA, K-means, SOM)."
    )
    story.append(Paragraph(conc_text, body_style))

    doc.build(story)
    print(f"Write-up PDF generated successfully at {pdf_path}")

if __name__ == '__main__':
    generate_writeup_pdf()

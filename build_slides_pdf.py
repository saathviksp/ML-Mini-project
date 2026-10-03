import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_slides_pdf(pdf_path='docs/slides_teammate1.pdf'):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    # Landscape slides (11in x 8.5in)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=10
    )
    
    category_style = ParagraphStyle(
        'SlideCat',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#DD6B20'),
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=8
    )

    story = []

    # Slide 1: Cover Slide
    story.append(Spacer(1, 100))
    cover_title = ParagraphStyle('CoverT', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=28, leading=34, textColor=colors.HexColor('#1A365D'), alignment=1)
    cover_sub = ParagraphStyle('CoverS', parent=styles['Normal'], fontName='Helvetica', fontSize=16, leading=20, textColor=colors.HexColor('#2B6CB0'), alignment=1)
    cover_tag = ParagraphStyle('CoverTag', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#DD6B20'), alignment=1)
    
    story.append(Paragraph("UE24CS352A Machine Learning • Mini-Project Assignment", cover_tag))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Pulsar Candidate Classification", cover_title))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Teammate 1: Supervised Learning Pipeline", cover_sub))
    story.append(Spacer(1, 20))
    story.append(Paragraph("GDA Baseline (Scratch) | Random Forest (Sklearn & Scratch) | Threshold Tuning", cover_tag))
    story.append(PageBreak())

    # Slide 2: Problem Statement
    story.append(Paragraph("TEAMMATE 1 • SUPERVISED CLASSIFICATION", category_style))
    story.append(Paragraph("Problem Statement & HTRU2 Dataset Overview", title_style))
    story.append(Paragraph("• <b>Astronomical Challenge:</b> Radio pulsar search surveys produce millions of candidate signals, mostly noise/RFI.", body_style))
    story.append(Paragraph("• <b>Dataset:</b> HTRU2 contains 17,898 candidate signals with 8 continuous features derived from integrated profiles and DM-SNR curves.", body_style))
    story.append(Paragraph("• <b>Class Imbalance:</b> 16,259 non-pulsars (90.84%) vs 1,639 true pulsars (9.16%). Ratio is 1:9.9.", body_style))
    story.append(Paragraph("• <b>Goal:</b> High-recall automated classification to ensure true pulsar signals are not discarded.", body_style))
    if os.path.exists('outputs/plots/class_distribution.png'):
        story.append(Spacer(1, 10))
        story.append(RLImage('outputs/plots/class_distribution.png', width=380, height=220))
    story.append(PageBreak())

    # Slide 3: Mathematical Formulations
    story.append(Paragraph("TEAMMATE 1 • SUPERVISED CLASSIFICATION", category_style))
    story.append(Paragraph("Supervised Methods & Mathematical Formulations", title_style))
    story.append(Paragraph("<b>1. Gaussian Discriminant Analysis (GDA Baseline from Scratch):</b>", body_style))
    story.append(Paragraph("&nbsp;&nbsp;• Prior: &phi; = (1/N) &Sigma; 1{y<sup>(i)</sup> = 1}", body_style))
    story.append(Paragraph("&nbsp;&nbsp;• Means: &mu;<sub>0</sub> = &Sigma; 1{y<sup>(i)</sup>=0} x<sup>(i)</sup> / N<sub>0</sub>, &nbsp;&nbsp; &mu;<sub>1</sub> = &Sigma; 1{y<sup>(i)</sup>=1} x<sup>(i)</sup> / N<sub>1</sub>", body_style))
    story.append(Paragraph("&nbsp;&nbsp;• Shared Covariance: &Sigma; = (1/N) &Sigma; (x<sup>(i)</sup> - &mu;<sub>y<sup>(i)</sup></sub>)(x<sup>(i)</sup> - &mu;<sub>y<sup>(i)</sup></sub>)<sup>T</sup>", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>2. Random Forest (Scikit-Learn & From Scratch):</b>", body_style))
    story.append(Paragraph("&nbsp;&nbsp;• Entropy: E(S) = -&Sigma; p(x) log<sub>2</sub> p(x)", body_style))
    story.append(Paragraph("&nbsp;&nbsp;• Information Gain: IG = E(parent) - (N<sub>left</sub>/N) E(left) - (N<sub>right</sub>/N) E(right)", body_style))
    story.append(Paragraph("&nbsp;&nbsp;• Ensemble of decision trees trained on bootstrap samples with random feature sub-selection (&radic;p features).", body_style))
    story.append(PageBreak())

    # Slide 4: Results Table
    story.append(Paragraph("TEAMMATE 1 • SUPERVISED CLASSIFICATION", category_style))
    story.append(Paragraph("Empirical Results & Model Comparison", title_style))
    
    table_data = [
        ['Model Architecture', 'Threshold', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'ROC AUC'],
        ['GDA Baseline (Scratch)', '0.50', '97.91%', '89.63%', '87.76%', '98.74%', '0.9708'],
        ['GDA Baseline (Opt Th.)', '0.39', '97.57%', '90.85%', '83.94%', '98.25%', '0.9708'],
        ['RF (Scikit-Learn)', '0.50', '97.88%', '88.41%', '88.41%', '98.83%', '0.9678'],
        ['RF (Scikit-Learn Tuned)', '0.20', '96.76%', '91.16%', '77.46%', '97.32%', '0.9678'],
        ['RF From Scratch', '0.50', '97.21%', '90.85%', '80.98%', '97.85%', '0.9752'],
        ['RF From Scratch (Th. 0.2)', '0.20', '90.95%', '93.29%', '50.33%', '90.71%', '0.9752']
    ]
    t = Table(table_data, colWidths=[180, 75, 75, 75, 75, 80, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    story.append(t)
    story.append(PageBreak())

    # Slide 5: Threshold Sweep & Curves
    story.append(Paragraph("TEAMMATE 1 • SUPERVISED CLASSIFICATION", category_style))
    story.append(Paragraph("Threshold Sweep & Diagnostic Curves", title_style))
    if os.path.exists('outputs/plots/rf_threshold_sweep.png') and os.path.exists('outputs/plots/roc_curve.png'):
        img_table = Table([
            [RLImage('outputs/plots/rf_threshold_sweep.png', width=340, height=230), RLImage('outputs/plots/roc_curve.png', width=340, height=230)]
        ], colWidths=[360, 360])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(img_table)
    story.append(PageBreak())

    # Slide 6: Conclusions
    story.append(Paragraph("TEAMMATE 1 • SUPERVISED CLASSIFICATION", category_style))
    story.append(Paragraph("Summary & Teammate 2 Handoff", title_style))
    story.append(Paragraph("• <b>Rigorous Pipeline:</b> 5-fold CV with upsampling in training folds ONLY ensured data leakage was completely prevented.", body_style))
    story.append(Paragraph("• <b>Baseline & Models:</b> Custom GDA baseline achieved 97.91% accuracy and 89.63% recall. Tuned Random Forest achieved 0.9678 ROC-AUC.", body_style))
    story.append(Paragraph("• <b>Threshold Tuning:</b> Moving decision boundary from 0.5 to ~0.2 increased recall to 91.16% - 93.29%, fulfilling paper recommendations.", body_style))
    story.append(Paragraph("• <b>Handoff to Teammate 2:</b> 1,639 true pulsar candidate instances are formatted and ready for unsupervised PCA, K-means, and SOM.", body_style))

    doc.build(story)
    print(f"Slides PDF generated successfully at {pdf_path}")

if __name__ == '__main__':
    generate_slides_pdf()

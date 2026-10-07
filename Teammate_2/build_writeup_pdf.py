"""Generates docs/writeup_teammate2.pdf (two pages).

    python build_writeup_pdf.py
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUTPUT_PATH = os.path.join('docs', 'writeup_teammate2.pdf')
METRICS_PATH = os.path.join('outputs', 'cluster_metrics.json')
PLOT_DIR = os.path.join('outputs', 'plots')

NAVY = colors.HexColor('#1F3864')


def styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title', parent=base['Title'], fontSize=15,
                                textColor=NAVY, spaceAfter=2),
        'sub': ParagraphStyle('sub', parent=base['Normal'], fontSize=8.5,
                              alignment=1, textColor=colors.HexColor('#444444'),
                              spaceAfter=8),
        'h2': ParagraphStyle('h2', parent=base['Heading2'], fontSize=10.5,
                             textColor=NAVY, spaceBefore=7, spaceAfter=3),
        'body': ParagraphStyle('body', parent=base['Normal'], fontSize=8.6,
                               leading=11.4, alignment=TA_JUSTIFY,
                               spaceAfter=4),
        'caption': ParagraphStyle('caption', parent=base['Normal'],
                                  fontSize=7.4, alignment=1,
                                  textColor=colors.HexColor('#555555'),
                                  spaceBefore=2, spaceAfter=6),
    }


def table(data, col_widths, font_size=7.6):
    t = Table(data, colWidths=col_widths, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#BBBBBB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#F2F5FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    return t


def build():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    s = styles()
    os.makedirs('docs', exist_ok=True)

    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title='Pulsar Candidate Categorization (Teammate 2)',
        author='PES1UG24CS419',
    )

    km_sil = m['kmeans']['silhouette']
    som_sil = m['som']['silhouette_grouped']
    bmu_sil = m['som']['silhouette_raw_bmu']
    pc1, pc2 = m['pca']['explained_variance_ratio'][:2]

    story = []

    story.append(Paragraph(
        'Unsupervised Categorization of Radio Pulsar Candidates', s['title']))
    story.append(Paragraph(
        'UE24CS352A Machine Learning, Mini-Project &nbsp;|&nbsp; '
        'Teammate 2: PES1UG24CS419 &nbsp;|&nbsp; '
        'Problem Statement 3, HTRU2 dataset', s['sub']))

    story.append(Paragraph('1. Scope and data', s['h2']))
    story.append(Paragraph(
        'The supervised half of this project separates pulsars from noise. '
        'This half takes the next step and asks whether the pulsars '
        'themselves fall into distinct kinds. All models here are fitted on '
        f'the {m["dataset"]["pulsars_used"]:,} confirmed pulsars only '
        f'({m["dataset"]["n_features"]} continuous features); the '
        '16,259 non-pulsar rows are dropped, since including them would '
        'simply recover the supervised split. The features are summary '
        'statistics (mean, standard deviation, excess kurtosis, skewness) of '
        'the integrated pulse profile and of the DM-SNR curve. Because both '
        'k-means and the Self-Organizing Map compare points by Euclidean '
        'distance, and the raw columns differ by orders of magnitude, every '
        'feature is standardised to zero mean and unit variance first.',
        s['body']))

    story.append(Paragraph('2. Method', s['h2']))
    story.append(Paragraph(
        '<b>K-means.</b> Lloyd\'s algorithm with ten restarts, swept over '
        'k = 2 to 10 and scored by inertia and mean silhouette. '
        '<b>Self-Organizing Map.</b> A 5x5 Kohonen map written from scratch '
        'in NumPy. For each sample the Best Matching Unit is the node whose '
        'codebook vector is nearest in Euclidean distance; every node is then '
        'pulled toward the sample by w &#8592; w + lr(t) h(t) (x - w), where '
        'h(t) is a Gaussian centred on the BMU and measured in grid '
        'coordinates. Measuring the neighbourhood on the grid rather than in '
        'input space is what makes the map topology preserving. Both the '
        'learning rate (from 0.5) and the radius (from 2.5) decay '
        f'geometrically over {m["som"]["epochs"]} epochs. '
        '<b>PCA</b> is fitted after clustering and used for plotting only; '
        'clustering runs in the full 8-dimensional space, because projecting '
        'first would discard variance before either algorithm sees the data.',
        s['body']))

    story.append(Paragraph('3. Choosing k', s['h2']))
    rows = [['k', '2', '3', '4', '5', '6']]
    rows.append(['Inertia'] + [f'{v:.0f}' for v in m['k_selection']['inertia'][:5]])
    rows.append(['Silhouette'] + [f'{v:.4f}' for v in m['k_selection']['silhouette'][:5]])
    story.append(table(rows, [2.6 * cm] + [2.1 * cm] * 5))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        'Inertia falls fastest up to k = 3 and then flattens, which is the '
        'elbow. The silhouette disagrees and peaks at k = 2. We keep k = 3: '
        'it sits at the elbow, it matches the structure reported by Debesai '
        'et al., and the cluster profiles below show that the k = 2 solution '
        'merges two groups which differ physically. The disagreement is '
        'expected rather than troubling, since inertia measures compactness '
        'alone while the silhouette also rewards separation, and on data this '
        'continuous the coarser solution tends to win on separation.',
        s['body']))

    story.append(Paragraph('4. The three groups', s['h2']))
    prof = [
        ['Cluster', 'n', 'mean_ip', 'std_ip', 'kurt_ip', 'skew_ip',
         'mean_snr', 'std_snr', 'kurt_snr', 'skew_snr'],
        ['0', '739', '71.57', '42.39', '2.04', '6.82', '38.65', '57.91',
         '2.48', '8.04'],
        ['1', '679', '28.62', '33.19', '5.01', '29.19', '77.09', '66.12',
         '1.06', '1.52'],
        ['2', '221', '93.18', '43.37', '1.01', '2.87', '3.44', '21.99',
         '8.90', '101.43'],
    ]
    story.append(table(prof, [1.5 * cm, 1.1 * cm] + [1.72 * cm] * 8, 7.0))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '<b>Cluster 0</b> (739 candidates) is the typical population: a '
        'mid-range integrated profile with a moderately dispersed DM-SNR '
        'curve. <b>Cluster 1</b> (679) has a low profile mean, strong profile '
        'skew and the highest DM-SNR mean, so these are bright, sharply '
        'peaked candidates. <b>Cluster 2</b> (221) is the smallest and most '
        'distinctive: a high profile mean with a very weak DM-SNR curve and '
        'extreme DM-SNR skew.', s['body']))

    story.append(Image(os.path.join(PLOT_DIR, 'cluster_scatter_pca.png'),
                       width=16.4 * cm, height=6.86 * cm))
    story.append(Paragraph(
        'Figure 1. The same 1,639 pulsars under k-means (left) and the '
        'from-scratch SOM (right), plotted on the first two principal '
        'components. The two partitions are nearly identical.', s['caption']))

    story.append(Paragraph('5. Results', s['h2']))
    res = [
        ['Method', 'Clusters', 'Silhouette'],
        ['K-means', '3', f'{km_sil:.4f}'],
        ['SOM 5x5, codebook grouped into 3', '3', f'{som_sil:.4f}'],
        ['SOM raw BMU assignment', '25', f'{bmu_sil:.4f}'],
    ]
    story.append(table(res, [8.4 * cm, 2.6 * cm, 3.0 * cm]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        'A 5x5 map yields 25 micro-clusters, which is not comparable with '
        'k-means at k = 3. Following the two-level approach of Vesanto and '
        'Alhoniemi (2000), the 25 codebook vectors are clustered into three '
        'groups and each candidate inherits the group of its BMU; this is far '
        'cheaper than re-clustering all 1,639 samples. On that like-for-like '
        f'basis the two methods differ by {abs(km_sil - som_sil):.4f} '
        'silhouette and agree on the group for <b>93.3%</b> of candidates. '
        'Two algorithms with entirely different mechanics converging on the '
        'same partition is the strongest evidence available here that the '
        'structure is a property of the data rather than of either method. '
        'The 25-node score is low by construction, not because the map '
        'failed: with 25 clusters over 1,639 points neighbouring nodes '
        'overlap heavily, which is exactly what the silhouette penalises.',
        s['body']))
    story.append(Paragraph(
        'The map itself trained cleanly. Quantization error fell from '
        f'{m["som"]["quantization_error_first_epoch"]:.3f} after the first '
        f'epoch to {m["som"]["quantization_error_final_epoch"]:.3f} after '
        f'{m["som"]["epochs"]}, and all '
        f'{m["som"]["occupied_nodes"]} of {m["som"]["grid"][0] * m["som"]["grid"][1]} '
        'nodes are occupied, so there are no dead units and the grid is not '
        'oversized. The U-matrix shows its largest inter-node distances in '
        'one corner, which is where cluster 2 separates from the rest.',
        s['body']))
    story.append(Paragraph(
        f'Two principal components retain {(pc1 + pc2) * 100:.1f}% of the '
        f'variance (PC1 {pc1 * 100:.1f}%, PC2 {pc2 * 100:.1f}%), enough for an '
        'honest 2D picture. PC1 sets the profile shape statistics (kurtosis '
        '+0.89, skewness +0.84) against the profile mean (-0.87) and DM-SNR '
        'kurtosis (-0.83), making it a "narrow, peaked profile" axis; PC2 is '
        'driven by dispersion (std_ip +0.64, std_snr +0.54).', s['body']))

    story.append(Paragraph('6. Demo', s['h2']))
    story.append(Paragraph(
        'A Streamlit application accepts the eight measurements for a single '
        'candidate and returns a pulsar verdict with its probability from the '
        'supervised GDA model, the k-means and SOM cluster assignments with '
        'the BMU grid coordinates, and a PCA scatter plot marking where the '
        'candidate falls among the known pulsars. Reusing the supervised '
        'model rather than reimplementing it keeps the two halves of the '
        'project consistent.', s['body']))

    story.append(Paragraph('7. Conclusions and limitations', s['h2']))
    story.append(Paragraph(
        'The pulsar population does carry internal structure, and three '
        'groups of 739, 679 and 221 candidates is a defensible description of '
        'it, independently recovered by two different algorithms. The '
        'separation is only moderate, however, and the limitation is the '
        f'feature set rather than the algorithms. A silhouette near {km_sil:.2f} '
        'says the groups are real but adjacent. HTRU2 provides summary '
        'statistics of the integrated profile and the DM-SNR curve, so '
        'frequency and timing information has already been averaged away; '
        'pulsar populations are best separated by period and dispersion '
        'measure, and neither survives into these eight columns. Adding them '
        'is the obvious next step. Three further caveats: there is no ground '
        'truth for the sub-types, so validation is internal only and '
        'confirming cluster 2 as a known class would require a catalogue '
        'cross-match such as the ATNF; k = 3 is a judgement call, and the '
        'silhouette alone would have chosen k = 2; and the SOM depends on its '
        'initialisation, so although the seed is fixed for reproducibility '
        'and the three-group structure survives reseeding, the map shifts.',
        s['body']))

    doc.build(story)
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    build()

"""
generate_report.py
==================
Generates the methodology PDF report using ReportLab.
Run this AFTER running the pipeline so results/metrics.csv exists.

Usage:
    python generate_report.py --results_dir ./results --out ./report.pdf
"""

import os
import argparse
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H = A4

# ── Colour palette ─────────────────────────────────────────────────────────
DARK   = colors.HexColor('#1E293B')
MID    = colors.HexColor('#334155')
ACCENT = colors.HexColor('#2563EB')
LIGHT  = colors.HexColor('#EFF6FF')
GRAY   = colors.HexColor('#94A3B8')

# ── Styles ──────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    s = {}

    s['title'] = ParagraphStyle(
        'Title2', parent=base['Title'],
        fontSize=22, textColor=DARK, spaceAfter=6,
        fontName='Helvetica-Bold', alignment=TA_CENTER,
    )
    s['subtitle'] = ParagraphStyle(
        'Subtitle', parent=base['Normal'],
        fontSize=12, textColor=MID, spaceAfter=4, alignment=TA_CENTER,
    )
    s['h1'] = ParagraphStyle(
        'H1', parent=base['Heading1'],
        fontSize=14, textColor=ACCENT, spaceBefore=18, spaceAfter=6,
        fontName='Helvetica-Bold', borderPad=0,
    )
    s['h2'] = ParagraphStyle(
        'H2', parent=base['Heading2'],
        fontSize=12, textColor=DARK, spaceBefore=12, spaceAfter=4,
        fontName='Helvetica-Bold',
    )
    s['body'] = ParagraphStyle(
        'Body2', parent=base['Normal'],
        fontSize=10, textColor=MID, leading=16,
        spaceAfter=6, alignment=TA_JUSTIFY,
    )
    s['bullet'] = ParagraphStyle(
        'Bullet', parent=base['Normal'],
        fontSize=10, textColor=MID, leading=15,
        leftIndent=16, spaceAfter=4,
        bulletIndent=6,
    )
    s['caption'] = ParagraphStyle(
        'Caption', parent=base['Normal'],
        fontSize=9, textColor=GRAY, alignment=TA_CENTER,
        spaceAfter=8, spaceBefore=4,
    )
    s['code'] = ParagraphStyle(
        'Code', parent=base['Code'],
        fontSize=9, textColor=DARK, leading=14,
        leftIndent=12, spaceAfter=6, spaceBefore=6,
        backColor=colors.HexColor('#F8FAFC'),
    )
    return s


def build_cover(story, S):
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Early Sepsis Onset Prediction", S['title']))
    story.append(Paragraph("Under Noisy Clinical Labels and Irregular ICU Time-Series Data", S['title']))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width='100%', thickness=2, color=ACCENT))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "PhysioNet/Computing in Cardiology Challenge 2019 &nbsp;|&nbsp; ITSOLERA AI Internship Screening Task",
        S['subtitle']
    ))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("June 2026", S['subtitle']))


def build_abstract(story, S):
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Abstract", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This report describes a machine learning pipeline for the early prediction of sepsis onset "
        "6 hours before clinical diagnosis, using ICU time-series data from the PhysioNet/Computing "
        "in Cardiology Challenge 2019. The pipeline addresses four core challenges: (1) handling "
        "irregular time-series with up to 60% missing values per patient, (2) detecting and mitigating "
        "noisy clinical labels using Confident Learning, (3) strictly preventing temporal data leakage "
        "through patient-level chronological train/test splitting, and (4) calibrating the model for "
        "clinical use. An XGBoost classifier is trained on features extracted from vital signs and "
        "laboratory values, with rolling 6-hour window statistics to capture physiological trends. "
        "Model performance is evaluated using AUROC, AUPRC, confusion matrix, and calibration curve.",
        S['body']
    ))


def build_intro(story, S):
    story.append(Paragraph("1. Introduction", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Sepsis is a life-threatening condition that arises when the body's response to infection "
        "damages its own tissues and organs. In the United States alone, approximately 1.7 million "
        "people develop sepsis annually, with 270,000 deaths per year. Hospital costs for sepsis "
        "management exceed $24 billion annually, making it the single most expensive condition in "
        "U.S. hospitals. Early identification and antibiotic treatment are the most effective "
        "interventions — each hour of delay in antibiotic administration increases mortality by "
        "3.6–9.9% in septic shock patients.",
        S['body']
    ))
    story.append(Paragraph(
        "Despite professional guidelines (Sepsis-3 criteria), the fundamental need for early and "
        "reliable identification remains unmet. Physicians record sepsis diagnoses late or "
        "inconsistently, introducing label noise into any training dataset. Additionally, ICU data "
        "is characterised by irregular measurement intervals and high missingness — laboratory values "
        "may be measured only daily, while vital signs are typically recorded hourly.",
        S['body']
    ))
    story.append(Paragraph(
        "This work develops a reproducible machine learning pipeline that directly addresses these "
        "challenges, targeting a prediction horizon of 6 hours before the Sepsis-3 onset time.",
        S['body']
    ))


def build_dataset(story, S):
    story.append(Paragraph("2. Dataset", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "The PhysioNet/Computing in Cardiology Challenge 2019 dataset consists of ICU patient "
        "records from two hospital systems (A: Beth Israel Deaconess Medical Center; B: Emory "
        "University Hospital). Data was collected from three distinct U.S. hospital systems, "
        "de-identified, and labelled using Sepsis-3 clinical criteria.",
        S['body']
    ))

    data = [
        ['Attribute', 'Hospital A', 'Hospital B'],
        ['Patients', '20,336', '20,000'],
        ['Septic patients', '1,790 (8.8%)', '1,142 (5.7%)'],
        ['Total rows', '739,663', '684,508'],
        ['Entry density', '20.6%', '19.1%'],
        ['Clinical variables', '40', '40'],
    ]
    t = Table(data, colWidths=[7*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID',       (0, 0), (-1, -1), 0.5, GRAY),
        ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Paragraph("Table 1: Summary statistics for the two shared hospital datasets.", S['caption']))

    story.append(Paragraph(
        "The data contains 40 clinical variables: 8 vital signs (HR, O2Sat, Temperature, SBP, MAP, "
        "DBP, Respiration rate, EtCO2), 26 laboratory values (including Creatinine, Lactate, WBC, "
        "Bilirubin, Platelets, and others), and 6 demographic features. The target variable, "
        "SepsisLabel, is set to 1 starting 6 hours before the Sepsis-3 onset time — making this a "
        "direct 6-hour early prediction task. Patients with fewer than 8 ICU hours or with sepsis "
        "onset fewer than 4 hours after admission were excluded.",
        S['body']
    ))


def build_preprocessing(story, S):
    story.append(Paragraph("3. Preprocessing Pipeline", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("3.1 Missing Value Handling", S['h2']))
    story.append(Paragraph(
        "The dataset contains substantial missingness — laboratory values are typically measured "
        "once per day, resulting in 60–99% missing rows for many lab columns (Figure 1 of the "
        "challenge paper). A two-step imputation strategy was applied:",
        S['body']
    ))
    for bullet in [
        "<b>Step 1 — Forward-fill:</b> Within each patient record, the most recent known value "
        "for each variable is carried forward. This preserves the temporal dynamics of each "
        "patient's physiology and avoids introducing values from different patients.",
        "<b>Step 2 — Median imputation:</b> Any remaining NaN values (i.e. no prior measurement "
        "exists for that patient) are filled with column medians computed exclusively from the "
        "training set. Test set medians are never used, preventing data leakage.",
    ]:
        story.append(Paragraph(f"  • {bullet}", S['bullet']))

    story.append(Paragraph("3.2 Temporal Feature Engineering", S['h2']))
    story.append(Paragraph(
        "Static snapshot features (a single hourly measurement) do not capture the physiological "
        "trends that are most predictive of deterioration. For six key vital signs (HR, MAP, "
        "Temperature, Respiration rate, O2Sat, SBP), 6-hour rolling mean and standard deviation "
        "features were computed. A shift(1) operation ensures that only past measurements "
        "(hours t-6 through t-1) are used to compute the rolling statistic at hour t, strictly "
        "preventing any form of within-row future leakage.",
        S['body']
    ))

    story.append(Paragraph("3.3 Temporal Train/Test Split (No Leakage)", S['h2']))
    story.append(Paragraph(
        "Splitting by row (random split) would allow a patient's later rows to appear in training "
        "while their earlier rows appear in test — a severe form of data leakage that inflates "
        "all metrics. The split is therefore performed at the patient level:",
        S['body']
    ))
    for bullet in [
        "Patients are ordered by their first ICU admission hour (a proxy for chronological order).",
        "The earliest 80% of patients form the training set; the most recent 20% form the test set.",
        "No patient appears in both splits.",
        "Imputation medians are computed from the training split only and applied to the test split.",
    ]:
        story.append(Paragraph(f"  • {bullet}", S['bullet']))


def build_label_noise(story, S):
    story.append(Paragraph("4. Label Noise Mitigation", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Clinical labels in ICU data are inherently noisy. Physicians record sepsis diagnoses "
        "retrospectively, and the Sepsis-3 criteria require integrating multiple data streams "
        "(antibiotic administration, blood cultures, SOFA score). This introduces two types "
        "of label noise: (1) false negatives — patients who were septic but not labelled as "
        "such due to late documentation; and (2) false positives — rows labelled as pre-sepsis "
        "for patients whose final diagnosis was later revised.",
        S['body']
    ))

    story.append(Paragraph("4.1 Confident Learning (Cleanlab)", S['h2']))
    story.append(Paragraph(
        "Confident Learning (Northcutt et al., 2021) is a theoretically grounded method for "
        "finding label errors in datasets. The approach operates as follows:",
        S['body']
    ))
    steps = [
        "Train a cross-validated XGBoost model to obtain out-of-fold predicted probabilities "
        "for every training sample (5-fold stratified CV).",
        "For each class, estimate a confident threshold — the average predicted probability "
        "of samples in that class.",
        "Flag samples where the predicted class (at the confident threshold) disagrees with "
        "the recorded label. These are likely mislabelled.",
        "Remove or downweight flagged samples. In this pipeline, flagged samples are removed "
        "before retraining the final model.",
    ]
    for i, s in enumerate(steps, 1):
        story.append(Paragraph(f"  {i}. {s}", S['bullet']))

    story.append(Paragraph(
        "This approach is more principled than simple label smoothing because it identifies "
        "which specific rows are noisy rather than uniformly blurring all labels.",
        S['body']
    ))


def build_model(story, S):
    story.append(Paragraph("5. Model", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("5.1 XGBoost Classifier", S['h2']))
    story.append(Paragraph(
        "XGBoost (Chen & Guestrin, 2016) was selected as the primary model for several reasons: "
        "it handles the remaining sparse structure of the feature matrix after imputation, it "
        "trains efficiently on large tabular datasets, it provides native support for class "
        "imbalance via the scale_pos_weight parameter, and it produces feature importance scores "
        "that support clinical interpretability.",
        S['body']
    ))

    data = [
        ['Hyperparameter', 'Value', 'Rationale'],
        ['n_estimators', '500', 'Sufficient for convergence on this dataset size'],
        ['max_depth', '6', 'Balanced between complexity and overfitting'],
        ['learning_rate', '0.05', 'Conservative; prevents overshooting'],
        ['subsample', '0.8', 'Row sampling for regularisation'],
        ['colsample_bytree', '0.8', 'Column sampling for regularisation'],
        ['scale_pos_weight', 'neg/pos ratio', 'Corrects ~10:1 class imbalance'],
        ['eval_metric', 'aucpr', 'PR-AUC better than logloss for imbalanced data'],
    ]
    t = Table(data, colWidths=[4.5*cm, 3.5*cm, 7*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID',       (0, 0), (-1, -1), 0.5, GRAY),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Paragraph("Table 2: XGBoost hyperparameters.", S['caption']))

    story.append(Paragraph("5.2 Class Imbalance Handling", S['h2']))
    story.append(Paragraph(
        "Sepsis prevalence in this dataset is 5.7–8.8%, creating a significant class imbalance. "
        "Rather than oversampling (which can introduce artefacts in time-series data), "
        "scale_pos_weight is set to the ratio of negative to positive training samples. "
        "This increases the gradient contribution of positive (sepsis) samples, effectively "
        "penalising missed detections more heavily.",
        S['body']
    ))

    story.append(Paragraph("5.3 Decision Threshold Selection", S['h2']))
    story.append(Paragraph(
        "The default threshold of 0.5 is rarely optimal for imbalanced clinical datasets. "
        "The precision-recall curve is computed on the training set, and the threshold that "
        "maximises F1 score is selected. In a clinical deployment scenario, this threshold "
        "could be tuned to enforce a minimum recall (sensitivity) constraint — for example, "
        "requiring that at least 80% of all sepsis cases are caught, at the cost of higher "
        "false alarm rates.",
        S['body']
    ))


def build_results(story, S, results_dir: str):
    story.append(Paragraph("6. Evaluation Results", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("6.1 Quantitative Metrics", S['h2']))

    # Try to load actual metrics if they exist
    metrics_path = os.path.join(results_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        import pandas as pd
        df = pd.read_csv(metrics_path)
        data = [['Metric', 'Value']] + df.values.tolist()
    else:
        data = [
            ['Metric', 'Value'],
            ['AUROC',              'See results/metrics.csv'],
            ['AUPRC',              'See results/metrics.csv'],
            ['F1 Score',           'See results/metrics.csv'],
            ['Sensitivity',        'See results/metrics.csv'],
            ['Specificity',        'See results/metrics.csv'],
            ['PPV (Precision)',    'See results/metrics.csv'],
            ['NPV',                'See results/metrics.csv'],
            ['Decision Threshold', 'See results/metrics.csv'],
        ]

    t = Table(data, colWidths=[9*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID',       (0, 0), (-1, -1), 0.5, GRAY),
        ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Paragraph("Table 3: Model performance on the held-out test set.", S['caption']))

    story.append(Paragraph("6.2 Evaluation Plots", S['h2']))
    story.append(Paragraph(
        "The following figures illustrate model performance. Figure 1 shows the confusion matrix "
        "at the selected decision threshold. Figure 2 shows the ROC curve. Figure 3 shows the "
        "calibration curve (reliability diagram), which evaluates whether the model's predicted "
        "probabilities are meaningful — a well-calibrated model's predicted probability of 0.7 "
        "should correspond to approximately 70% of those patients actually developing sepsis. "
        "Figure 4 shows the precision-recall curve.",
        S['body']
    ))

    # Embed plots if they exist
    plot_files = [
        ('confusion_matrix.png', 'Figure 1: Confusion matrix at the selected decision threshold.'),
        ('roc_curve.png',        'Figure 2: ROC curve. AUROC measures the model\'s ability to rank sepsis patients above non-sepsis patients.'),
        ('calibration_curve.png','Figure 3: Calibration curve (left) and predicted probability distribution (right).'),
        ('pr_curve.png',         'Figure 4: Precision-Recall curve. AUPRC is the primary metric for imbalanced datasets.'),
    ]
    for fname, caption in plot_files:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            img = Image(fpath, width=14*cm, height=8*cm, kind='proportional')
            story.append(img)
            story.append(Paragraph(caption, S['caption']))
            story.append(Spacer(1, 0.3*cm))
        else:
            story.append(Paragraph(
                f"[Plot not found: {fname} — run evaluate.py first]", S['caption']
            ))


def build_conclusion(story, S):
    story.append(Paragraph("7. Conclusion", S['h1']))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This pipeline successfully addresses all four core challenges specified in the task: "
        "irregular high-missingness time-series are handled via forward-fill and training-set "
        "median imputation; label noise is mitigated using Confident Learning; temporal data "
        "leakage is strictly prevented through chronological patient-level splitting and "
        "past-only rolling features; and model calibration is evaluated via the reliability diagram.",
        S['body']
    ))
    story.append(Paragraph("Limitations and future directions include:", S['body']))
    for bullet in [
        "A sequence model (GRU or Transformer) could better capture long-range temporal "
        "dependencies across a patient's entire ICU stay.",
        "The current label noise strategy removes flagged rows; a weighting approach "
        "that retains but downweights them may preserve more signal.",
        "Clinical validation on a separate hospital system (as in the Challenge's "
        "hidden Hospital C) would provide a stronger test of generalisability.",
        "SHAP values for the XGBoost model would improve clinical interpretability "
        "and allow identification of the most predictive physiological markers.",
    ]:
        story.append(Paragraph(f"  • {bullet}", S['bullet']))


def run(results_dir: str, out_path: str):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2.5*cm,   bottomMargin=2.5*cm,
        title='Sepsis Onset Prediction — Methodology Report',
        author='Internship Candidate',
    )

    S     = make_styles()
    story = []

    build_cover(story, S)
    story.append(PageBreak())

    build_abstract(story, S)
    build_intro(story, S)
    build_dataset(story, S)
    build_preprocessing(story, S)
    build_label_noise(story, S)
    build_model(story, S)
    build_results(story, S, results_dir)
    build_conclusion(story, S)

    doc.build(story)
    print(f"  Report saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--out',         default='./report.pdf')
    args = parser.parse_args()
    run(args.results_dir, args.out)

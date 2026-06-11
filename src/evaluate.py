"""
evaluate.py
===========
Loads trained model and test set, generates all required evaluation outputs:
  - Confusion matrix
  - ROC curve + AUROC
  - Calibration curve (reliability diagram)
  - Precision-recall curve + AUPRC
  - Full metrics table

Usage:
    python evaluate.py --processed_dir ./processed --model_dir ./models --out_dir ./results
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, classification_report,
)
from sklearn.calibration import calibration_curve

# ── Consistent plot style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.dpi'   : 150,
})
BLUE  = '#2563EB'
RED   = '#DC2626'
GREEN = '#16A34A'
GRAY  = '#6B7280'


def plot_confusion_matrix(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Sepsis', 'Sepsis'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix', fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_roc_curve(y_true, y_proba, auroc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], '--', color=GRAY, lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate (1 – Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve', fontweight='bold', pad=12)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_calibration_curve(y_true, y_proba, out_path):
    """
    Reliability diagram: if the model says 0.7 probability,
    do ~70% of those patients actually develop sepsis?
    A perfectly calibrated model sits on the diagonal.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: reliability diagram
    ax = axes[0]
    ax.plot(prob_pred, prob_true, 's-', color=BLUE, lw=2,
            markersize=6, label='XGBoost')
    ax.plot([0, 1], [0, 1], '--', color=GRAY, lw=1.5, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Right: histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_proba[y_true == 0], bins=40, alpha=0.6,
             color=BLUE, label='No Sepsis', density=True)
    ax2.hist(y_proba[y_true == 1], bins=40, alpha=0.6,
             color=RED,  label='Sepsis',    density=True)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Predicted Probability Distribution', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_pr_curve(y_true, y_proba, auprc, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()   # no-skill classifier
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color=GREEN, lw=2, label=f'AUPRC = {auprc:.4f}')
    ax.axhline(baseline, linestyle='--', color=GRAY, lw=1,
               label=f'No-skill baseline ({baseline:.3f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('Precision-Recall Curve', fontweight='bold', pad=12)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_metrics_table(y_true, y_pred, y_proba, threshold, out_path):
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)
    npv         = tn / max(tn + fn, 1)
    f1          = f1_score(y_true, y_pred)

    metrics = {
        'Metric' : ['AUROC', 'AUPRC', 'F1 Score', 'Sensitivity (Recall)',
                    'Specificity', 'PPV (Precision)', 'NPV',
                    'True Positives', 'False Positives',
                    'False Negatives', 'True Negatives',
                    'Decision Threshold'],
        'Value'  : [f'{auroc:.4f}', f'{auprc:.4f}', f'{f1:.4f}',
                    f'{sensitivity:.4f}', f'{specificity:.4f}',
                    f'{ppv:.4f}', f'{npv:.4f}',
                    str(tp), str(fp), str(fn), str(tn),
                    f'{threshold:.4f}'],
    }
    df = pd.DataFrame(metrics)
    df.to_csv(out_path, index=False)

    print("\n  ── Evaluation Metrics ──────────────────────────────")
    for _, row in df.iterrows():
        print(f"  {row['Metric']:<30} {row['Value']}")
    print(f"  Saved → {out_path}")


def run(processed_dir: str, model_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/3] Loading model and test data …")
    with open(os.path.join(model_dir, 'xgb_model.pkl'), 'rb') as f:
        artefacts = pickle.load(f)

    model     = artefacts['model']
    threshold = artefacts['threshold']
    feat_cols = artefacts['feat_cols']

    test      = pd.read_parquet(os.path.join(processed_dir, 'test.parquet'))
    X_test    = test[feat_cols].values
    y_test    = test['SepsisLabel'].values.astype(int)
    y_proba   = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_proba >= threshold).astype(int)

    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    print(f"  Test samples: {len(y_test):,} | Positives: {y_test.sum():,}")

    print("\n[2/3] Generating plots …")
    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(out_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_proba, auroc,
                   os.path.join(out_dir, 'roc_curve.png'))
    plot_calibration_curve(y_test, y_proba,
                           os.path.join(out_dir, 'calibration_curve.png'))
    plot_pr_curve(y_test, y_proba, auprc,
                  os.path.join(out_dir, 'pr_curve.png'))

    print("\n[3/3] Saving metrics …")
    save_metrics_table(y_test, y_pred, y_proba, threshold,
                       os.path.join(out_dir, 'metrics.csv'))

    print(f"\n  All results saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='./processed')
    parser.add_argument('--model_dir',     default='./models')
    parser.add_argument('--out_dir',       default='./results')
    args = parser.parse_args()
    run(args.processed_dir, args.model_dir, args.out_dir)

"""
train.py
========
Trains an XGBoost classifier on the cleaned training data.
Handles class imbalance via scale_pos_weight.
Selects optimal threshold using precision-recall curve.
Saves the trained model for evaluation.

Usage:
    python train.py --processed_dir ./processed --model_dir ./models
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score
)

BASE_FEATURES = [
    'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2',
    'BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN',
    'Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct',
    'Glucose','Lactate','Magnesium','Phosphate','Potassium',
    'Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
    'Fibrinogen','Platelets',
    'Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS',
    'HR_roll6_mean','HR_roll6_std',
    'MAP_roll6_mean','MAP_roll6_std',
    'Temp_roll6_mean','Temp_roll6_std',
    'Resp_roll6_mean','Resp_roll6_std',
    'O2Sat_roll6_mean','O2Sat_roll6_std',
    'SBP_roll6_mean','SBP_roll6_std',
]


def select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1 score on training data.
    Clinical note: you may want to bias toward higher recall (catch more sepsis)
    by choosing the threshold where recall >= 0.80 instead.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # avoid divide by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])   # last element has no threshold
    best_threshold = thresholds[best_idx]
    print(f"  Optimal threshold: {best_threshold:.4f}  "
          f"(F1={f1_scores[best_idx]:.4f}, "
          f"Precision={precisions[best_idx]:.4f}, "
          f"Recall={recalls[best_idx]:.4f})")
    return float(best_threshold)


def run(processed_dir: str, model_dir: str, use_clean: bool = True):
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    train_file = 'train_clean.parquet' if use_clean else 'train.parquet'
    train = pd.read_parquet(os.path.join(processed_dir, train_file))
    test  = pd.read_parquet(os.path.join(processed_dir, 'test.parquet'))

    feat_cols = [c for c in BASE_FEATURES if c in train.columns]

    X_train = train[feat_cols].values
    y_train = train['SepsisLabel'].values.astype(int)
    X_test  = test[feat_cols].values
    y_test  = test['SepsisLabel'].values.astype(int)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / max(pos, 1)   # scale_pos_weight handles class imbalance
    print(f"  Train  — rows: {len(X_train):,}  "
          f"| positives: {pos:,} ({pos/len(y_train)*100:.1f}%)  "
          f"| scale_pos_weight: {spw:.1f}")
    print(f"  Test   — rows: {len(X_test):,}  "
          f"| positives: {(y_test==1).sum():,} "
          f"({(y_test==1).mean()*100:.1f}%)")

    # ── Train XGBoost ──────────────────────────────────────────────────────
    print("\n[2/4] Training XGBoost …")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric='aucpr',       # area under PR curve — better for imbalance
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluate on training set to pick threshold ─────────────────────────
    print("\n[3/4] Selecting decision threshold …")
    train_proba = model.predict_proba(X_train)[:, 1]
    threshold   = select_threshold(y_train, train_proba)

    # ── Quick test-set sanity check ────────────────────────────────────────
    print("\n[4/4] Test-set quick check …")
    test_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, test_proba)
    auprc = average_precision_score(y_test, test_proba)
    y_pred = (test_proba >= threshold).astype(int)
    f1    = f1_score(y_test, y_pred)
    print(f"  AUROC : {auroc:.4f}")
    print(f"  AUPRC : {auprc:.4f}")
    print(f"  F1    : {f1:.4f}  (at threshold {threshold:.4f})")

    # ── Save ───────────────────────────────────────────────────────────────
    artefacts = {
        'model'      : model,
        'threshold'  : threshold,
        'feat_cols'  : feat_cols,
        'train_auroc': auroc,
    }
    out_path = os.path.join(model_dir, 'xgb_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(artefacts, f)
    print(f"\n  Model saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='./processed')
    parser.add_argument('--model_dir',     default='./models')
    parser.add_argument('--no_clean', action='store_true',
                        help='Use original labels instead of cleanlab-cleaned ones')
    args = parser.parse_args()
    run(args.processed_dir, args.model_dir, use_clean=not args.no_clean)

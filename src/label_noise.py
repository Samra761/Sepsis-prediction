"""
label_noise.py
==============
Detects and mitigates label noise in the sepsis training set
using Confident Learning (cleanlab library).

Why this works for sepsis:
  Physicians often record sepsis late. Cleanlab finds rows where
  the model is confident of a label but the recorded label disagrees
  — these are likely mislabeled (e.g. actually septic but labelled 0).

Usage:
    python label_noise.py --processed_dir ./processed
"""

import os
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from cleanlab.filter import find_label_issues

# Same feature list as preprocess.py (plus rolling features)
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


def detect_label_issues(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """
    Step 1: Get out-of-fold predicted probabilities via cross-validation.
    Step 2: Pass probabilities + original labels to cleanlab.
    Returns boolean mask — True where a label is likely noisy.
    """
    print(f"  Running {n_folds}-fold cross-validation for OOF probabilities …")

    # Fast XGBoost for OOF predictions (not the final model)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')

    print("  Running cleanlab Confident Learning …")
    label_issues = find_label_issues(
        labels=y,
        pred_probs=oof_probs,
        return_indices_ranked_by='self_confidence',
    )

    # Convert indices to boolean mask
    noise_mask = np.zeros(len(y), dtype=bool)
    noise_mask[label_issues] = True

    n_issues = noise_mask.sum()
    pct = n_issues / len(y) * 100
    print(f"  Cleanlab flagged {n_issues:,} rows as likely noisy ({pct:.1f}% of training set)")
    print(f"    → of these, {noise_mask[y==1].sum():,} are label=1 flagged as noisy")
    print(f"    → of these, {noise_mask[y==0].sum():,} are label=0 flagged as noisy")

    return noise_mask


def run(processed_dir: str):
    print("\n[1/3] Loading training data …")
    train = pd.read_parquet(os.path.join(processed_dir, 'train.parquet'))

    # Only keep columns that exist in the dataframe
    feat_cols = [c for c in BASE_FEATURES if c in train.columns]
    X = train[feat_cols].values
    y = train['SepsisLabel'].values.astype(int)

    print(f"  Rows: {len(X):,}  |  Positives: {y.sum():,} ({y.mean()*100:.1f}%)")

    print("\n[2/3] Detecting label noise …")
    noise_mask = detect_label_issues(X, y)

    print("\n[3/3] Saving cleaned training set …")
    train['is_noisy'] = noise_mask

    # Strategy: REMOVE flagged rows (conservative — you can also downweight)
    train_clean = train[~noise_mask].copy()
    train_noisy = train[noise_mask].copy()

    train_clean.to_parquet(os.path.join(processed_dir, 'train_clean.parquet'), index=False)
    train_noisy.to_parquet(os.path.join(processed_dir, 'train_noisy.parquet'), index=False)

    print(f"  Saved train_clean.parquet  ({len(train_clean):,} rows)")
    print(f"  Saved train_noisy.parquet  ({len(train_noisy):,} rows — for reference)")
    print(f"  New positive rate after cleaning: "
          f"{train_clean['SepsisLabel'].mean()*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='./processed')
    args = parser.parse_args()
    run(args.processed_dir)

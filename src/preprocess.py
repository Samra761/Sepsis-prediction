"""
preprocess.py
=============
Loads the PhysioNet Sepsis Challenge 2019 .psv files,
handles missing values, and creates the final feature matrix.

Usage:
    python preprocess.py --data_dir /path/to/training  --out_dir ./processed
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ── Column names exactly as in the challenge files ──────────────────────────
VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

LAB_COLS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]

DEMO_COLS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

FEATURE_COLS = VITAL_COLS + LAB_COLS + DEMO_COLS
TARGET_COL   = 'SepsisLabel'


def load_all_patients(data_dir: str) -> pd.DataFrame:
    """Read every .psv file in data_dir and stack into one DataFrame."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.psv')))
    if not files:
        raise FileNotFoundError(f"No .psv files found in {data_dir}")

    print(f"  Found {len(files)} patient files …")
    dfs = []
    for f in files:
        pid = os.path.splitext(os.path.basename(f))[0]   # e.g. "p000001"
        df  = pd.read_csv(f, sep='|')
        df.insert(0, 'patient_id', pid)
        df['hour'] = np.arange(len(df))                   # 0-based hour index
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(combined):,}  |  "
          f"Septic patients: {combined.groupby('patient_id')[TARGET_COL].max().sum():,}")
    return combined


def impute(df: pd.DataFrame, train_medians: pd.Series = None):
    """
    Two-step imputation (no leakage):
      1. Forward-fill within each patient (carry last known value forward).
      2. Fill remaining NaNs with TRAINING SET medians.

    If train_medians is None we compute them (call this on train split only).
    Returns (imputed_df, medians_series).
    """
    feature_df = df[FEATURE_COLS].copy()

    # Step 1 – forward-fill within patient
    feature_df = (df[['patient_id'] + FEATURE_COLS]
                    .groupby('patient_id')[FEATURE_COLS]
                    .transform(lambda s: s.ffill()))

    # Step 2 – fill remaining with training medians
    if train_medians is None:
        train_medians = feature_df.median()

    feature_df = feature_df.fillna(train_medians)

    return feature_df, train_medians


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6-hour rolling mean and std for vital signs.
    Groups by patient so windows don't bleed across patients.
    NOTE: shift(1) ensures we only use PAST values (no leakage at current hour).
    """
    roll_feats = []
    key_vitals = ['HR', 'MAP', 'Temp', 'Resp', 'O2Sat', 'SBP']

    for col in key_vitals:
        grp = df.groupby('patient_id')[col]
        # shift(1) so hour t uses hours t-6 .. t-1  →  strict past only
        roll_feats.append(
            grp.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
               .rename(f'{col}_roll6_mean')
        )
        roll_feats.append(
            grp.transform(lambda s: s.shift(1).rolling(6, min_periods=1).std().fillna(0))
               .rename(f'{col}_roll6_std')
        )

    return pd.concat([df] + roll_feats, axis=1)


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Split by PATIENT ID so no patient appears in both train and test.
    Patients are sorted by their first ICU hour (earliest admitted → training).
    This prevents temporal leakage.
    """
    patient_ids = df['patient_id'].unique()

    # sort patients by when they first appear (proxy for admission time)
    first_hour = (df.groupby('patient_id')['hour'].min()
                    .sort_values()
                    .index.tolist())

    n_test  = int(len(first_hour) * test_size)
    test_ids  = set(first_hour[-n_test:])          # later-admitted patients → test
    train_ids = set(first_hour[:-n_test])

    train = df[df['patient_id'].isin(train_ids)].copy()
    test  = df[df['patient_id'].isin(test_ids)].copy()

    print(f"  Train: {len(train_ids):,} patients | "
          f"Test: {len(test_ids):,} patients")
    return train, test


def build_feature_matrix(df_raw: pd.DataFrame,
                          train_medians: pd.Series = None,
                          fit: bool = True):
    """
    Full preprocessing pipeline for one split.
    fit=True  → compute medians from this split (training)
    fit=False → use provided train_medians (test split, no leakage)
    """
    feature_df, medians = impute(df_raw, train_medians if not fit else None)

    # Attach meta columns back
    feature_df['patient_id'] = df_raw['patient_id'].values
    feature_df['hour']       = df_raw['hour'].values
    feature_df[TARGET_COL]   = df_raw[TARGET_COL].values

    # Rolling features
    feature_df = add_rolling_features(feature_df)

    return feature_df, medians


def run(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/4] Loading patient files …")
    df_all = load_all_patients(data_dir)

    print("\n[2/4] Temporal train/test split …")
    train_raw, test_raw = temporal_train_test_split(df_all)

    print("\n[3/4] Preprocessing train split …")
    train_df, train_medians = build_feature_matrix(train_raw, fit=True)

    print("\n[4/4] Preprocessing test split (using train medians) …")
    test_df, _ = build_feature_matrix(test_raw,
                                       train_medians=train_medians,
                                       fit=False)

    # Save
    train_df.to_parquet(os.path.join(out_dir, 'train.parquet'), index=False)
    test_df.to_parquet(os.path.join(out_dir,  'test.parquet'),  index=False)
    train_medians.to_csv(os.path.join(out_dir, 'train_medians.csv'))

    print(f"\n  Saved to {out_dir}/")
    print(f"  Train rows: {len(train_df):,}  "
          f"| Positive rate: {train_df[TARGET_COL].mean()*100:.1f}%")
    print(f"  Test rows:  {len(test_df):,}  "
          f"| Positive rate: {test_df[TARGET_COL].mean()*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Folder containing .psv patient files')
    parser.add_argument('--out_dir', default='./processed',
                        help='Where to save processed parquet files')
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)

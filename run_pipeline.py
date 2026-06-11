"""
run_pipeline.py
===============
Runs the entire pipeline end-to-end with one command.

Usage:
    python run_pipeline.py --data_dir /path/to/training_setA

The data_dir should contain the .psv patient files downloaded from PhysioNet.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import src.preprocess  as preprocess
import src.label_noise as label_noise
import src.train       as train
import src.evaluate    as evaluate


def main():
    parser = argparse.ArgumentParser(
        description='Sepsis Onset Prediction — full pipeline'
    )
    parser.add_argument('--data_dir',      required=True,
                        help='Folder containing .psv patient files')
    parser.add_argument('--processed_dir', default='./processed')
    parser.add_argument('--model_dir',     default='./models')
    parser.add_argument('--out_dir',       default='./results')
    parser.add_argument('--skip_noise',    action='store_true',
                        help='Skip cleanlab step (faster, for testing)')
    args = parser.parse_args()

    print("=" * 60)
    print("  SEPSIS ONSET PREDICTION — PIPELINE START")
    print("=" * 60)

    print("\n── STEP 1: Preprocessing ──")
    preprocess.run(args.data_dir, args.processed_dir)

    if not args.skip_noise:
        print("\n── STEP 2: Label Noise Detection ──")
        label_noise.run(args.processed_dir)
    else:
        print("\n── STEP 2: Label Noise — SKIPPED ──")
        # Fall back to raw training file
        import shutil, os as _os
        src = _os.path.join(args.processed_dir, 'train.parquet')
        dst = _os.path.join(args.processed_dir, 'train_clean.parquet')
        if not _os.path.exists(dst):
            shutil.copy(src, dst)

    print("\n── STEP 3: Model Training ──")
    train.run(args.processed_dir, args.model_dir)

    print("\n── STEP 4: Evaluation ──")
    evaluate.run(args.processed_dir, args.model_dir, args.out_dir)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {args.out_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

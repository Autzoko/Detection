"""
Run nnDetection inference on training set cases.

Usage:
    # Predict on all training cases
    python scripts/predict_train.py 100 RetinaUNetV001 --fold 0

    # Predict on 10 random training cases
    python scripts/predict_train.py 100 RetinaUNetV001 --fold 0 -n 10

    # Predict on specific cases
    python scripts/predict_train.py 100 RetinaUNetV001 --fold 0 --cases case_00000 case_00005

How it works:
    1. Reads splits.pkl from the training directory
    2. Adds a "test" key with the selected training case IDs
    3. Saves updated splits.pkl
    4. Calls nndet_predict with --test_split --no_preprocess
    5. Restores original splits.pkl
"""

import argparse
import os
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

from nndet.io import get_task, get_training_dir


def main():
    parser = argparse.ArgumentParser(description="Predict on training set cases")
    parser.add_argument("task", type=str, help="Task id e.g. 100")
    parser.add_argument("model", type=str, help="Model name e.g. RetinaUNetV001")
    parser.add_argument("-f", "--fold", type=int, default=0, help="Fold (default: 0)")
    parser.add_argument("-n", "--num_samples", type=int, default=None,
                        help="Number of training samples to predict on (default: all)")
    parser.add_argument("--cases", type=str, nargs="+", default=None,
                        help="Specific case IDs to predict on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "all"],
                        help="Which split to predict on (default: train)")
    parser.add_argument("-o", "--overwrites", type=str, nargs="+", default=None,
                        help="Overwrites for nndet_predict")
    args = parser.parse_args()

    # Resolve paths
    task_name = get_task(args.task, name=True)
    task_model_dir = Path(os.getenv("det_models"))
    training_dir = get_training_dir(task_model_dir / task_name / args.model, args.fold)
    splits_path = training_dir / "splits.pkl"

    # Load splits
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    # Select cases
    if args.cases is not None:
        case_ids = args.cases
        print(f"Using {len(case_ids)} specified cases")
    else:
        if args.split == "train":
            case_ids = sorted(splits[0]["train"])
        elif args.split == "val":
            case_ids = sorted(splits[0]["val"])
        else:  # all
            case_ids = sorted(splits[0]["train"] + splits[0]["val"])

        if args.num_samples is not None and args.num_samples < len(case_ids):
            random.seed(args.seed)
            case_ids = sorted(random.sample(case_ids, args.num_samples))
            print(f"Sampled {len(case_ids)} cases from {args.split} split")
        else:
            print(f"Using all {len(case_ids)} cases from {args.split} split")

    print(f"Cases: {case_ids[:5]}{'...' if len(case_ids) > 5 else ''}")

    # Backup splits.pkl and add "test" key
    backup_path = splits_path.with_suffix(".pkl.bak")
    shutil.copy2(splits_path, backup_path)
    print(f"Backed up splits.pkl to {backup_path}")

    try:
        splits[0]["test"] = case_ids
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"Added 'test' key with {len(case_ids)} cases to splits.pkl")

        # Build nndet_predict command
        cmd = [
            "nndet_predict", args.task, args.model,
            "-f", str(args.fold),
            "--test_split", "--no_preprocess",
        ]
        if args.overwrites:
            cmd.extend(["-o"] + args.overwrites)

        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nnndet_predict exited with code {result.returncode}")
            sys.exit(result.returncode)

    finally:
        # Restore original splits.pkl
        shutil.move(str(backup_path), str(splits_path))
        print(f"\nRestored original splits.pkl")


if __name__ == "__main__":
    main()

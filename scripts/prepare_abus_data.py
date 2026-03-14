"""
Prepare the ABUS (Automated Breast Ultrasound) public dataset for nnDetection.

Source data
-----------
  /Volumes/Autzoko/Dataset/Ultrasound/ABUS/data/
    Train/      (100 cases)
    Validation/ (30 cases)
    Test/       (70 cases)
  Each split contains:
    DATA/DATA_{id:03d}.nrrd     - 3D ultrasound volume (uint8)
    MASK/MASK_{id:03d}.nrrd     - Binary segmentation mask (uint8, 0/1)
    labels.csv                  - case_id, label (M/B), paths
    bbx_labels.csv              - id, c_x, c_y, c_z, len_x, len_y, len_z

Labels: M = malignant, B = benign (binary classification).
Each case has exactly one lesion.

All 200 cases go into imagesTr/labelsTr for training.
A small validation set (15%) is split out for early stopping only.
No separate test set — all data used for training.

Output: nnDetection Task format
-------------------------------
  Task200_ABUS/raw_splitted/
    imagesTr/  case_XXXXX_0000.nii.gz   (all 200 cases)
    labelsTr/  case_XXXXX.nii.gz + case_XXXXX.json
  dataset.json

Classes: 0 = Benign, 1 = Malignant
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# Config (defaults, overridable via CLI)
# ============================================================
DEFAULT_ABUS_ROOT = "/Volumes/Autzoko/Dataset/Ultrasound/ABUS/data"
DEFAULT_OUTPUT_ROOT = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/ABUS"
TASK_NAME = "Task200_ABUS"

# Class mapping: M=malignant=1, B=benign=0
CLASS_MAP = {"B": 0, "M": 1}


def read_split(split_dir):
    """Read labels.csv and return list of case dicts."""
    labels_csv = split_dir / "labels.csv"
    cases = []
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "case_id": int(row["case_id"]),
                "label": row["label"],  # M or B
                "class_id": CLASS_MAP[row["label"]],
                "data_path": split_dir / "DATA" / f"DATA_{int(row['case_id']):03d}.nrrd",
                "mask_path": split_dir / "MASK" / f"MASK_{int(row['case_id']):03d}.nrrd",
            })
    return cases


def convert_case(case, out_case_id, images_dir, labels_dir):
    """Convert one ABUS case to nnDetection format.

    - Image: save as NIfTI with preserved spacing
    - Label: instance segmentation mask (single instance per case) + JSON
    """
    case_name = f"case_{out_case_id:05d}"

    # Read image
    img_sitk = sitk.ReadImage(str(case["data_path"]))
    arr = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)

    # Save image as NIfTI
    img_out = sitk.GetImageFromArray(arr.astype(np.float32))
    img_out.SetSpacing(img_sitk.GetSpacing())
    img_out.SetOrigin(img_sitk.GetOrigin())
    img_out.SetDirection(img_sitk.GetDirection())
    sitk.WriteImage(img_out, str(images_dir / f"{case_name}_0000.nii.gz"))

    # Read mask
    mask_sitk = sitk.ReadImage(str(case["mask_path"]))
    mask_arr = sitk.GetArrayFromImage(mask_sitk)  # (z, y, x), uint8, 0/1

    # Convert binary mask to instance mask (single instance = 1)
    # Use connected components to handle potential multiple components
    instance_mask = np.zeros_like(mask_arr, dtype=np.float32)
    if mask_arr.max() > 0:
        # Label connected components
        labeled_sitk = sitk.ConnectedComponent(sitk.GetImageFromArray(mask_arr.astype(np.uint8)))
        labeled_arr = sitk.GetArrayFromImage(labeled_sitk)
        n_components = labeled_arr.max()

        if n_components == 1:
            instance_mask[mask_arr > 0] = 1
        else:
            # Multiple components — each gets its own instance ID
            # All share the same class (this case has one label: M or B)
            instance_mask = labeled_arr.astype(np.float32)

    # Save instance mask
    mask_out = sitk.GetImageFromArray(instance_mask)
    mask_out.SetSpacing(mask_sitk.GetSpacing())
    mask_out.SetOrigin(mask_sitk.GetOrigin())
    mask_out.SetDirection(mask_sitk.GetDirection())
    sitk.WriteImage(mask_out, str(labels_dir / f"{case_name}.nii.gz"))

    # Save JSON
    n_instances = int(instance_mask.max())
    instances = {}
    for i in range(1, n_instances + 1):
        instances[str(i)] = case["class_id"]

    json_path = labels_dir / f"{case_name}.json"
    with open(json_path, "w") as f:
        json.dump({"instances": instances}, f, indent=2)

    return {
        "case_name": case_name,
        "original_id": case["case_id"],
        "label": case["label"],
        "class_id": case["class_id"],
        "n_instances": n_instances,
        "shape": list(arr.shape),
        "spacing": list(img_sitk.GetSpacing()),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ABUS data for nnDetection")
    parser.add_argument("--abus_root", type=str, default=DEFAULT_ABUS_ROOT,
                        help="Path to ABUS data root (contains Train/Validation/Test)")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="Output root for nnDetection task")
    args = parser.parse_args()

    ABUS_ROOT = Path(args.abus_root)
    OUTPUT_ROOT = Path(args.output_root)

    task_dir = OUTPUT_ROOT / TASK_NAME
    raw_dir = task_dir / "raw_splitted"
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"

    for d in [images_tr, labels_tr]:
        d.mkdir(parents=True, exist_ok=True)

    # Read ALL splits
    train_cases = read_split(ABUS_ROOT / "Train")
    val_cases = read_split(ABUS_ROOT / "Validation")
    test_cases = read_split(ABUS_ROOT / "Test")

    all_cases = train_cases + val_cases + test_cases

    print(f"Train:      {len(train_cases)} cases "
          f"(M={sum(1 for c in train_cases if c['label']=='M')}, "
          f"B={sum(1 for c in train_cases if c['label']=='B')})")
    print(f"Validation: {len(val_cases)} cases "
          f"(M={sum(1 for c in val_cases if c['label']=='M')}, "
          f"B={sum(1 for c in val_cases if c['label']=='B')})")
    print(f"Test:       {len(test_cases)} cases "
          f"(M={sum(1 for c in test_cases if c['label']=='M')}, "
          f"B={sum(1 for c in test_cases if c['label']=='B')})")
    print(f"Total:      {len(all_cases)} cases")

    # ALL cases → imagesTr/labelsTr (no separate test set)
    # Small val set for early stopping only
    print(f"\nAll {len(all_cases)} cases → imagesTr/labelsTr")

    case_id = 0
    stats = []

    print("\nConverting all cases...")
    for case in all_cases:
        # Track which original split it came from
        if case in train_cases:
            src = "train"
        elif case in val_cases:
            src = "val"
        else:
            src = "test"

        info = convert_case(case, case_id, images_tr, labels_tr)
        info["source_split"] = src
        stats.append(info)
        case_id += 1
        if case_id % 20 == 0:
            print(f"  {case_id}/{len(all_cases)} done")

    print(f"  {len(all_cases)}/{len(all_cases)} done")

    # ---- Write dataset.json ----
    dataset_json = {
        "task": TASK_NAME,
        "name": "ABUS Breast Ultrasound Lesion Detection",
        "target_class": None,
        "test_labels": True,
        "labels": {
            "0": "Benign",
            "1": "Malignant",
        },
        "modalities": {
            "0": "US",
        },
        "dim": 3,
    }
    with open(task_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\nWrote dataset.json")

    # ---- Write splits_final.pkl ----
    # All cases used for training; 15% held out as val for early stopping
    import pickle
    from sklearn.model_selection import train_test_split

    all_names = [s["case_name"] for s in stats]
    all_labels = [s["label"] for s in stats]

    train_names, val_names = train_test_split(
        all_names, test_size=0.15, random_state=42, stratify=all_labels
    )

    splits = [{"train": train_names, "val": val_names}]
    splits_path = task_dir / "splits_final.pkl"
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Wrote splits_final.pkl (train={len(train_names)}, val={len(val_names)})")

    # ---- Write dataset_statistics.csv ----
    csv_path = task_dir / "dataset_statistics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_name", "original_id", "label", "class_id",
            "n_instances", "shape", "spacing", "source_split"
        ])
        writer.writeheader()
        writer.writerows(stats)
    print(f"Wrote dataset_statistics.csv")

    # ---- Summary ----
    total_m = sum(1 for s in stats if s["label"] == "M")
    total_b = sum(1 for s in stats if s["label"] == "B")
    total_inst = sum(s["n_instances"] for s in stats)
    print(f"\n{'=' * 60}")
    print(f"  ABUS Dataset Preparation Complete")
    print(f"{'=' * 60}")
    print(f"  Total cases: {len(stats)} (all in imagesTr)")
    print(f"  Malignant: {total_m}, Benign: {total_b}")
    print(f"  Total instances: {total_inst}")
    print(f"  nnDet split: train={len(train_names)}, val={len(val_names)}")
    print(f"  Output: {task_dir}")
    print(f"{'=' * 60}")

    # ---- Print what to do next ----
    print(f"""
NEXT STEPS:
-----------
1. Set environment variables:
   export det_data="{OUTPUT_ROOT}"
   export det_models="{OUTPUT_ROOT}/models"
   export OMP_NUM_THREADS=1

2. Run nnDetection planning + preprocessing:
   nndet_prep 200

3. Train:
   nndet_train 200 --fold 0

4. (Optional) Predict on training data for analysis:
   nndet_predict 200 --fold 0
""")


if __name__ == "__main__":
    main()

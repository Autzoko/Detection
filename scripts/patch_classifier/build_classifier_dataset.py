"""
Build training dataset for 3D patch classifier from nnDetection predictions.

For each predicted bbox:
  1. Match to GT using 3D IoU → label as TP or FP
  2. Crop 3D patch from volume (bbox + 50% padding), resize to 64x64x64
  3. Normalize: clip to [1st, 99th] percentile of parent volume, scale to [0, 1]

Also generates random negative patches from each volume.

Label assignment rules:
  - IoU >= 0.1 with any GT bbox → positive candidate
  - Among candidates matching the same GT bbox, only highest-IoU → positive; rest → negative
  - IoU < 0.1 with all GT → negative (false positive)

Usage:
    python build_classifier_dataset.py --config config.yaml
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml
from scipy.ndimage import zoom
from skimage.measure import regionprops


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def iou_3d(box1, box2):
    """3D IoU. Boxes in nnDetection format: [z_min, y_min, z_max, y_max, x_min, x_max]."""
    z1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x1 = max(box1[4], box2[4])
    z2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    x2 = min(box1[5], box2[5])

    inter = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    vol1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    vol2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def extract_gt_boxes(label_path):
    """Extract GT bboxes from instance segmentation NIfTI."""
    label_sitk = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_sitk).astype(np.int32)  # (z, y, x)

    if label_arr.max() == 0:
        return []

    gt_boxes = []
    for prop in regionprops(label_arr):
        bbox = prop.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        # nnDetection format: [z_min, y_min, z_max, y_max, x_min, x_max]
        gt_boxes.append([bbox[0], bbox[1], bbox[3], bbox[4], bbox[2], bbox[5]])
    return gt_boxes


def crop_and_resize_patch(volume, box, padding_fraction, target_size):
    """Crop 3D patch centered on box with padding, resize to target_size.

    Args:
        volume: 3D numpy array (z, y, x)
        box: [z_min, y_min, z_max, y_max, x_min, x_max]
        padding_fraction: fraction of box size to pad on each side
        target_size: [D, H, W] target dimensions

    Returns:
        3D numpy array of shape target_size
    """
    z_min, y_min, z_max, y_max, x_min, x_max = box

    # Box dimensions
    dz = z_max - z_min
    dy = y_max - y_min
    dx = x_max - x_min

    # Add padding
    pad_z = int(dz * padding_fraction)
    pad_y = int(dy * padding_fraction)
    pad_x = int(dx * padding_fraction)

    # Padded crop bounds
    cz0 = int(z_min) - pad_z
    cz1 = int(z_max) + pad_z
    cy0 = int(y_min) - pad_y
    cy1 = int(y_max) + pad_y
    cx0 = int(x_min) - pad_x
    cx1 = int(x_max) + pad_x

    # Ensure minimum size of 4 voxels per dim (avoid degenerate patches)
    for _ in range(3):
        if cz1 - cz0 < 4:
            cz0 -= 1; cz1 += 1
        if cy1 - cy0 < 4:
            cy0 -= 1; cy1 += 1
        if cx1 - cx0 < 4:
            cx0 -= 1; cx1 += 1

    # Clip to volume bounds with padding
    vz, vy, vx = volume.shape
    # Calculate how much we need to pad if crop exceeds volume
    pad_before = [max(0, -cz0), max(0, -cy0), max(0, -cx0)]
    pad_after = [max(0, cz1 - vz), max(0, cy1 - vy), max(0, cx1 - vx)]

    # Clip crop coords to volume
    cz0_c = max(0, cz0)
    cz1_c = min(vz, cz1)
    cy0_c = max(0, cy0)
    cy1_c = min(vy, cy1)
    cx0_c = max(0, cx0)
    cx1_c = min(vx, cx1)

    patch = volume[cz0_c:cz1_c, cy0_c:cy1_c, cx0_c:cx1_c]

    # Pad if needed
    if any(p > 0 for p in pad_before + pad_after):
        patch = np.pad(patch, list(zip(pad_before, pad_after)), mode="constant", constant_values=0)

    # Resize to target size
    if patch.shape[0] == 0 or patch.shape[1] == 0 or patch.shape[2] == 0:
        return np.zeros(target_size, dtype=np.float32)

    zoom_factors = [t / s for t, s in zip(target_size, patch.shape)]
    patch_resized = zoom(patch.astype(np.float32), zoom_factors, order=1)

    # Handle potential rounding issues
    if patch_resized.shape != tuple(target_size):
        result = np.zeros(target_size, dtype=np.float32)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(patch_resized.shape, target_size))
        result[slices] = patch_resized[slices]
        return result

    return patch_resized


def normalize_patch(patch, p_low, p_high, volume=None):
    """Clip to percentiles and scale to [0, 1].

    If volume is provided, use volume-level percentiles.
    Otherwise use patch-level percentiles.
    """
    if volume is not None:
        vmin = np.percentile(volume, p_low)
        vmax = np.percentile(volume, p_high)
    else:
        vmin = np.percentile(patch, p_low)
        vmax = np.percentile(patch, p_high)

    if vmax <= vmin:
        return np.zeros_like(patch, dtype=np.float32)

    patch = np.clip(patch, vmin, vmax)
    patch = (patch - vmin) / (vmax - vmin)
    return patch.astype(np.float32)


def sample_random_negative(volume, gt_boxes, box_size_range, iou_threshold, max_attempts=50):
    """Sample a random bbox that doesn't overlap with any GT box."""
    vz, vy, vx = volume.shape

    for _ in range(max_attempts):
        # Random box size (within range of typical prediction sizes)
        dz = random.randint(box_size_range[0][0], box_size_range[0][1])
        dy = random.randint(box_size_range[1][0], box_size_range[1][1])
        dx = random.randint(box_size_range[2][0], box_size_range[2][1])

        # Random position
        z0 = random.randint(0, max(0, vz - dz))
        y0 = random.randint(0, max(0, vy - dy))
        x0 = random.randint(0, max(0, vx - dx))

        box = [z0, y0, z0 + dz, y0 + dy, x0, x0 + dx]

        # Check IoU with all GT boxes
        overlaps = [iou_3d(box, gt) for gt in gt_boxes] if gt_boxes else [0.0]
        if max(overlaps) < iou_threshold:
            return box

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print statistics without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    ds_cfg = cfg["dataset"]

    pred_dir = Path(paths["predictions_dir"])
    images_dir = Path(paths["images_dir"])
    labels_dir = Path(paths["labels_dir"])
    dataset_dir = Path(paths["dataset_dir"])

    target_size = ds_cfg["patch_size"]
    padding_frac = ds_cfg["padding_fraction"]
    p_low = ds_cfg["percentile_low"]
    p_high = ds_cfg["percentile_high"]
    min_score = ds_cfg["min_pred_score"]
    iou_thresh = ds_cfg["iou_match_threshold"]

    # Find all prediction files
    pred_files = sorted(pred_dir.glob("*_boxes.pkl"))
    case_ids = [f.stem.replace("_boxes", "") for f in pred_files]
    print(f"Found {len(case_ids)} cases with predictions")

    # All cases go into training; small val split only for early stopping / threshold tuning
    random.seed(ds_cfg["seed"])
    shuffled = list(case_ids)
    random.shuffle(shuffled)
    val_fraction = ds_cfg.get("val_fraction", 0.15)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_cases = set(shuffled[:n_val])
    # All cases are used for training (including val cases)
    train_cases = set(case_ids)

    print(f"Training on ALL {len(train_cases)} cases")
    print(f"Val subset ({len(val_cases)} cases) used only for early stopping / threshold tuning")
    print(f"Val cases: {sorted(val_cases)}")

    if not args.dry_run:
        for split in ["train", "val"]:
            (dataset_dir / split / "patches").mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        "train": {"tp": 0, "fp_pred": 0, "random_neg": 0, "gt_total": 0,
                  "duplicate_groups": 0},
        "val": {"tp": 0, "fp_pred": 0, "random_neg": 0, "gt_total": 0,
                "duplicate_groups": 0},
    }
    all_samples = {"train": [], "val": []}

    for case_idx, case_id in enumerate(case_ids):
        # All cases go into train; val cases are additionally copied to val split
        splits = ["train"]
        if case_id in val_cases:
            splits.append("val")
        split_label = "train+val" if case_id in val_cases else "train"
        print(f"\n[{case_idx+1}/{len(case_ids)}] {case_id} ({split_label})")

        # Load predictions
        with open(pred_dir / f"{case_id}_boxes.pkl", "rb") as f:
            pred = pickle.load(f)

        pred_boxes = pred["pred_boxes"]
        pred_scores = pred["pred_scores"]

        # Filter by minimum score
        score_mask = pred_scores >= min_score
        pred_boxes = pred_boxes[score_mask]
        pred_scores = pred_scores[score_mask]
        print(f"  Predictions: {len(pred_boxes)} (after score >= {min_score})")

        # Load volume
        vol_path = images_dir / f"{case_id}_0000.nii.gz"
        vol_sitk = sitk.ReadImage(str(vol_path))
        volume = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)  # (z, y, x)

        # Compute volume-level percentiles for normalization
        v_low = np.percentile(volume, p_low)
        v_high = np.percentile(volume, p_high)

        # Load GT
        gt_path = labels_dir / f"{case_id}.nii.gz"
        gt_boxes = extract_gt_boxes(gt_path) if gt_path.exists() else []
        print(f"  GT boxes: {len(gt_boxes)}")
        for split in splits:
            stats[split]["gt_total"] += len(gt_boxes)

        # Match predictions to GT
        pred_gt_matches = []
        for p_idx in range(len(pred_boxes)):
            best_iou = 0
            best_gt = -1
            for g_idx, gt_box in enumerate(gt_boxes):
                overlap = iou_3d(pred_boxes[p_idx], gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt = g_idx
            pred_gt_matches.append((p_idx, best_gt, best_iou))

        # Label assignment
        positive_candidates = [(p_idx, g_idx, iou_val)
                               for p_idx, g_idx, iou_val in pred_gt_matches
                               if iou_val >= iou_thresh]

        gt_to_best_pred = {}
        for p_idx, g_idx, iou_val in positive_candidates:
            if g_idx not in gt_to_best_pred or iou_val > gt_to_best_pred[g_idx][1]:
                gt_to_best_pred[g_idx] = (p_idx, iou_val)

        tp_pred_indices = set(p_idx for p_idx, _ in gt_to_best_pred.values())

        gt_pred_counts = {}
        for p_idx, g_idx, iou_val in positive_candidates:
            if g_idx not in gt_pred_counts:
                gt_pred_counts[g_idx] = 0
            gt_pred_counts[g_idx] += 1
        n_dup_groups = sum(1 for c in gt_pred_counts.values() if c > 1)
        for split in splits:
            stats[split]["duplicate_groups"] += n_dup_groups

        labels = {}
        for p_idx in range(len(pred_boxes)):
            if p_idx in tp_pred_indices:
                labels[p_idx] = 1
            else:
                labels[p_idx] = 0

        n_tp = sum(1 for v in labels.values() if v == 1)
        n_fp = sum(1 for v in labels.values() if v == 0)
        print(f"  Labels: {n_tp} TP, {n_fp} FP predictions")

        # Extract patches for all predictions
        for p_idx in range(len(pred_boxes)):
            box = pred_boxes[p_idx].tolist()
            score = float(pred_scores[p_idx])
            label = labels[p_idx]

            patch = crop_and_resize_patch(volume, box, padding_frac, target_size)
            patch = np.clip(patch, v_low, v_high)
            if v_high > v_low:
                patch = (patch - v_low) / (v_high - v_low)
            else:
                patch = np.zeros_like(patch)

            sample_name = f"{case_id}_pred{p_idx:04d}"
            sample_info = {
                "name": sample_name,
                "case_id": case_id,
                "pred_idx": p_idx,
                "box": box,
                "pred_score": score,
                "label": label,
                "source": "prediction",
            }

            # Save patch to all applicable splits
            for split in splits:
                if not args.dry_run:
                    np.save(dataset_dir / split / "patches" / f"{sample_name}.npy", patch)
                all_samples[split].append(sample_info)
                if label == 1:
                    stats[split]["tp"] += 1
                else:
                    stats[split]["fp_pred"] += 1

        # Generate random negative patches
        if gt_boxes:
            if len(pred_boxes) > 0:
                box_sizes_z = pred_boxes[:, 2] - pred_boxes[:, 0]
                box_sizes_y = pred_boxes[:, 3] - pred_boxes[:, 1]
                box_sizes_x = pred_boxes[:, 5] - pred_boxes[:, 4]
                size_range = [
                    [max(10, int(np.percentile(box_sizes_z, 10))),
                     int(np.percentile(box_sizes_z, 90))],
                    [max(5, int(np.percentile(box_sizes_y, 10))),
                     int(np.percentile(box_sizes_y, 90))],
                    [max(10, int(np.percentile(box_sizes_x, 10))),
                     int(np.percentile(box_sizes_x, 90))],
                ]
            else:
                size_range = [[20, 100], [10, 50], [30, 150]]

            n_random = ds_cfg["random_neg_per_volume"]
            for r_idx in range(n_random):
                rand_box = sample_random_negative(
                    volume, gt_boxes, size_range, iou_thresh)
                if rand_box is None:
                    continue

                patch = crop_and_resize_patch(volume, rand_box, padding_frac, target_size)
                patch = np.clip(patch, v_low, v_high)
                if v_high > v_low:
                    patch = (patch - v_low) / (v_high - v_low)
                else:
                    patch = np.zeros_like(patch)

                sample_name = f"{case_id}_randneg{r_idx:04d}"
                sample_info = {
                    "name": sample_name,
                    "case_id": case_id,
                    "pred_idx": -1,
                    "box": rand_box,
                    "pred_score": 0.0,
                    "label": 0,
                    "source": "random_negative",
                }

                for split in splits:
                    if not args.dry_run:
                        np.save(dataset_dir / split / "patches" / f"{sample_name}.npy", patch)
                    all_samples[split].append(sample_info)
                    stats[split]["random_neg"] += 1

    # Apply sampling ratios
    print("\n=== Applying Sample Ratios ===")
    for split in ["train", "val"]:
        samples = all_samples[split]
        tp_samples = [s for s in samples if s["label"] == 1]
        fp_samples = [s for s in samples if s["label"] == 0 and s["source"] == "prediction"]
        rn_samples = [s for s in samples if s["source"] == "random_negative"]

        n_tp = len(tp_samples)
        target_fp = int(n_tp * ds_cfg["fp_ratio"])
        target_rn = int(n_tp * ds_cfg["random_neg_ratio"])

        print(f"\n  {split}: {n_tp} TP, {len(fp_samples)} FP preds, {len(rn_samples)} rand neg")

        # Subsample FP and random negatives if needed
        if len(fp_samples) > target_fp:
            random.shuffle(fp_samples)
            fp_samples = fp_samples[:target_fp]
        if len(rn_samples) > target_rn:
            random.shuffle(rn_samples)
            rn_samples = rn_samples[:target_rn]

        # Oversample TPs if needed
        if ds_cfg.get("oversample_tp", False) and n_tp > 0:
            total_neg = len(fp_samples) + len(rn_samples)
            if total_neg > n_tp * 2:
                # Repeat TPs to get closer to 1:2 ratio
                repeat_factor = max(1, total_neg // (n_tp * 2))
                tp_samples = tp_samples * repeat_factor
                print(f"  Oversampled TPs by {repeat_factor}x → {len(tp_samples)}")

        balanced = tp_samples + fp_samples + rn_samples
        random.shuffle(balanced)

        print(f"  Final: {len(tp_samples)} TP, {len(fp_samples)} FP, "
              f"{len(rn_samples)} rand neg = {len(balanced)} total")

        all_samples[split] = balanced

    # Save sample manifests
    if not args.dry_run:
        for split in ["train", "val"]:
            manifest_path = dataset_dir / split / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(all_samples[split], f, indent=2)
            print(f"\n  Saved {split} manifest: {manifest_path}")

        # Save split info
        split_info = {
            "train_cases": sorted(train_cases),
            "val_cases": sorted(val_cases),
            "note": "All cases used for training; val is a subset for early stopping only",
            "stats": stats,
            "config": ds_cfg,
        }
        with open(dataset_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("=== Dataset Build Summary ===")
    print("=" * 60)
    for split in ["train", "val"]:
        s = stats[split]
        print(f"\n  {split}:")
        print(f"    GT lesions:       {s['gt_total']}")
        print(f"    TP predictions:   {s['tp']}")
        print(f"    FP predictions:   {s['fp_pred']}")
        print(f"    Random negatives: {s['random_neg']}")
        print(f"    Duplicate groups: {s['duplicate_groups']}")
        n_final = len(all_samples[split])
        n_pos = sum(1 for s in all_samples[split] if s["label"] == 1)
        print(f"    Final samples:    {n_final} ({n_pos} pos, {n_final - n_pos} neg)")


if __name__ == "__main__":
    main()

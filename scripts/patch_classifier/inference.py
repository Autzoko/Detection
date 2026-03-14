"""
Inference pipeline: apply trained 3D patch classifier to nnDetection predictions.

For a given volume and its nnDetection predictions:
  1. Crop and normalize each predicted bbox
  2. Run classifier → confidence score per candidate
  3. Group candidates by 3D IoU > 0.3 (duplicate clusters)
  4. Within each group, keep only highest classifier score
  5. Apply confidence threshold to filter FPs
  6. Output final kept bboxes with classifier scores

Also produces threshold tuning curves on the val set.

Usage:
    python inference.py --config config.yaml
    python inference.py --config config.yaml --threshold 0.5
    python inference.py --config config.yaml --tune  # sweep thresholds on val set
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from skimage.measure import regionprops

from model import PatchClassifier3D


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def iou_3d(box1, box2):
    """3D IoU. Boxes: [z_min, y_min, z_max, y_max, x_min, x_max]."""
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


def crop_and_normalize(volume, box, padding_fraction, target_size, v_low, v_high):
    """Crop, pad, resize, normalize a patch from volume."""
    from scipy.ndimage import zoom as scipy_zoom

    z_min, y_min, z_max, y_max, x_min, x_max = box
    dz, dy, dx = z_max - z_min, y_max - y_min, x_max - x_min
    pad_z = int(dz * padding_fraction)
    pad_y = int(dy * padding_fraction)
    pad_x = int(dx * padding_fraction)

    cz0 = int(z_min) - pad_z
    cz1 = int(z_max) + pad_z
    cy0 = int(y_min) - pad_y
    cy1 = int(y_max) + pad_y
    cx0 = int(x_min) - pad_x
    cx1 = int(x_max) + pad_x

    for _ in range(3):
        if cz1 - cz0 < 4: cz0 -= 1; cz1 += 1
        if cy1 - cy0 < 4: cy0 -= 1; cy1 += 1
        if cx1 - cx0 < 4: cx0 -= 1; cx1 += 1

    vz, vy, vx = volume.shape
    pad_before = [max(0, -cz0), max(0, -cy0), max(0, -cx0)]
    pad_after = [max(0, cz1 - vz), max(0, cy1 - vy), max(0, cx1 - vx)]

    patch = volume[max(0, cz0):min(vz, cz1), max(0, cy0):min(vy, cy1), max(0, cx0):min(vx, cx1)]
    if any(p > 0 for p in pad_before + pad_after):
        patch = np.pad(patch, list(zip(pad_before, pad_after)), mode="constant", constant_values=0)

    if 0 in patch.shape:
        return np.zeros(target_size, dtype=np.float32)

    zoom_factors = [t / s for t, s in zip(target_size, patch.shape)]
    patch = scipy_zoom(patch.astype(np.float32), zoom_factors, order=1)

    if patch.shape != tuple(target_size):
        result = np.zeros(target_size, dtype=np.float32)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(patch.shape, target_size))
        result[slices] = patch[slices]
        patch = result

    # Normalize
    patch = np.clip(patch, v_low, v_high)
    if v_high > v_low:
        patch = (patch - v_low) / (v_high - v_low)
    return patch.astype(np.float32)


def group_by_iou(boxes, scores, iou_threshold):
    """Group boxes into clusters where any pair has IoU > threshold.

    Returns list of groups, each group is list of (box_idx, box, score).
    """
    n = len(boxes)
    if n == 0:
        return []

    # Build adjacency
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if iou_3d(boxes[i], boxes[j]) > iou_threshold:
                union(i, j)

    # Collect groups
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return [g for g in groups.values()]


def resolve_duplicates(boxes, classifier_scores, pred_scores, iou_threshold):
    """Group candidates by IoU and keep highest classifier score per group.

    Returns:
        kept_indices: list of indices into original arrays
        group_info: list of dicts with group details
    """
    groups = group_by_iou(boxes, classifier_scores, iou_threshold)
    kept_indices = []
    group_info = []

    for group in groups:
        # Pick the one with highest classifier score
        best_idx = max(group, key=lambda i: classifier_scores[i])
        kept_indices.append(best_idx)
        group_info.append({
            "group_size": len(group),
            "kept_idx": best_idx,
            "kept_classifier_score": float(classifier_scores[best_idx]),
            "kept_pred_score": float(pred_scores[best_idx]),
            "all_indices": group,
        })

    return kept_indices, group_info


def extract_gt_boxes(label_path):
    """Extract GT bboxes from instance segmentation NIfTI."""
    label_sitk = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_sitk).astype(np.int32)
    if label_arr.max() == 0:
        return []
    gt_boxes = []
    for prop in regionprops(label_arr):
        bbox = prop.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        gt_boxes.append([bbox[0], bbox[1], bbox[3], bbox[4], bbox[2], bbox[5]])
    return gt_boxes


@torch.no_grad()
def classify_patches(model, patches, device, batch_size=32):
    """Run classifier on a list of patches, return probabilities."""
    model.eval()
    all_probs = []

    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i:i+batch_size])
        # (B, D, H, W) → (B, 1, D, H, W)
        batch = torch.from_numpy(batch[:, np.newaxis]).to(device)
        probs = model.predict_proba(batch)
        all_probs.extend(probs.cpu().numpy().flatten().tolist())

    return np.array(all_probs)


def process_case(model, volume, pred_boxes, pred_scores, cfg, device):
    """Process a single case: crop, classify, group, filter.

    Returns dict with results at multiple thresholds.
    """
    ds_cfg = cfg["dataset"]
    inf_cfg = cfg["inference"]
    target_size = ds_cfg["patch_size"]
    padding_frac = ds_cfg["padding_fraction"]
    p_low, p_high = ds_cfg["percentile_low"], ds_cfg["percentile_high"]

    # Volume-level percentiles
    v_low = np.percentile(volume, p_low)
    v_high = np.percentile(volume, p_high)

    # Filter by minimum prediction score
    min_score = cfg["inference"].get("min_pred_score", 0.05)
    mask = pred_scores >= min_score
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]

    if len(pred_boxes) == 0:
        return {
            "n_input": 0,
            "boxes": np.zeros((0, 6)),
            "pred_scores": np.zeros(0),
            "classifier_scores": np.zeros(0),
            "groups": [],
        }

    # Crop and normalize patches
    patches = []
    for box in pred_boxes:
        patch = crop_and_normalize(volume, box.tolist(), padding_frac, target_size, v_low, v_high)
        patches.append(patch)

    # Classify
    classifier_scores = classify_patches(model, patches, device)

    # Resolve duplicates
    dup_iou = inf_cfg["duplicate_iou_threshold"]
    kept_indices, group_info = resolve_duplicates(
        pred_boxes.tolist(), classifier_scores, pred_scores, dup_iou)

    return {
        "n_input": len(pred_boxes),
        "all_boxes": pred_boxes,
        "all_pred_scores": pred_scores,
        "all_classifier_scores": classifier_scores,
        "kept_indices": kept_indices,
        "kept_boxes": pred_boxes[kept_indices],
        "kept_pred_scores": pred_scores[kept_indices],
        "kept_classifier_scores": classifier_scores[kept_indices],
        "groups": group_info,
    }


def threshold_filter(result, threshold):
    """Apply confidence threshold to kept candidates."""
    if len(result["kept_indices"]) == 0:
        return np.zeros((0, 6)), np.zeros(0)

    mask = result["kept_classifier_scores"] >= threshold
    return result["kept_boxes"][mask], result["kept_classifier_scores"][mask]


def tune_threshold(all_results, all_gt_boxes, cfg):
    """Sweep thresholds and compute precision/recall/F1 at each."""
    inf_cfg = cfg["inference"]
    t_min, t_max = inf_cfg["threshold_range"]
    t_step = inf_cfg["threshold_step"]

    thresholds = np.arange(t_min, t_max + t_step, t_step)
    iou_thresh = cfg["inference"]["iou_match_threshold"]

    curves = []
    for t in thresholds:
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for case_id in all_gt_boxes:
            gt = all_gt_boxes[case_id]
            total_gt += len(gt)

            result = all_results[case_id]
            final_boxes, final_scores = threshold_filter(result, t)

            gt_matched = [False] * len(gt)
            for box in final_boxes:
                best_iou = 0
                best_g = -1
                for g_idx, gt_box in enumerate(gt):
                    if gt_matched[g_idx]:
                        continue
                    ov = iou_3d(box.tolist(), gt_box)
                    if ov > best_iou:
                        best_iou = ov
                        best_g = g_idx
                if best_iou >= iou_thresh and best_g >= 0:
                    total_tp += 1
                    gt_matched[best_g] = True
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        n_cases = len(all_gt_boxes)
        fp_per_case = total_fp / n_cases if n_cases > 0 else 0

        curves.append({
            "threshold": round(float(t), 3),
            "TP": int(total_tp),
            "FP": int(total_fp),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fp_per_case": round(fp_per_case, 2),
        })

    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override operating threshold")
    parser.add_argument("--tune", action="store_true",
                        help="Sweep thresholds and produce tuning curve")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Specific case IDs to process (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    inf_cfg = cfg["inference"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    ckpt_path = Path(paths["checkpoint_dir"]) / "best.pt"
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model_cfg = cfg["model"]
    model = PatchClassifier3D(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        num_blocks=model_cfg["num_blocks"],
        fc_hidden=model_cfg["fc_hidden"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded (best F1={checkpoint.get('best_f1', 'N/A')})")

    # Find cases
    pred_dir = Path(paths["test_predictions_dir"])
    images_dir = Path(paths["test_images_dir"])
    labels_dir = Path(paths["test_labels_dir"])
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        case_ids = args.cases
    else:
        pred_files = sorted(pred_dir.glob("*_boxes.pkl"))
        case_ids = [f.stem.replace("_boxes", "") for f in pred_files]

    # Load split info to identify val cases
    dataset_dir = Path(paths["dataset_dir"])
    split_info_path = dataset_dir / "split_info.json"
    if split_info_path.exists():
        with open(split_info_path) as f:
            split_info = json.load(f)
        val_cases = set(split_info["val_cases"])
    else:
        val_cases = set(case_ids)  # If no split info, treat all as val

    threshold = args.threshold or inf_cfg["operating_threshold"]

    # Process each case
    all_results = {}
    all_gt_boxes = {}

    for idx, case_id in enumerate(case_ids):
        print(f"\n[{idx+1}/{len(case_ids)}] {case_id}", end="")
        if case_id in val_cases:
            print(" (val)", end="")
        print()

        # Load predictions
        with open(pred_dir / f"{case_id}_boxes.pkl", "rb") as f:
            pred = pickle.load(f)

        # Load volume
        vol_path = images_dir / f"{case_id}_0000.nii.gz"
        vol_sitk = sitk.ReadImage(str(vol_path))
        volume = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)

        # Process
        result = process_case(model, volume, pred["pred_boxes"], pred["pred_scores"],
                              cfg, device)
        all_results[case_id] = result

        # Apply threshold
        final_boxes, final_scores = threshold_filter(result, threshold)

        # Load GT
        gt_path = labels_dir / f"{case_id}.nii.gz"
        if gt_path.exists():
            gt = extract_gt_boxes(gt_path)
            all_gt_boxes[case_id] = gt
        else:
            gt = []

        print(f"  Input: {result['n_input']} preds → "
              f"After dedup: {len(result['kept_indices'])} → "
              f"After threshold({threshold}): {len(final_boxes)}")
        if gt:
            print(f"  GT: {len(gt)} lesions")

        # Save per-case results
        case_output = {
            "pred_boxes": final_boxes,
            "pred_scores": final_scores,
            "pred_labels": np.zeros(len(final_boxes), dtype=np.int64),
        }
        with open(output_dir / f"{case_id}_boxes.pkl", "wb") as f:
            pickle.dump(case_output, f)

    # Threshold tuning on val set
    if args.tune and all_gt_boxes:
        val_results = {k: v for k, v in all_results.items() if k in val_cases}
        val_gt = {k: v for k, v in all_gt_boxes.items() if k in val_cases}

        if val_results and val_gt:
            print("\n" + "=" * 60)
            print("=== Threshold Tuning (Val Set) ===")
            print("=" * 60)

            curves = tune_threshold(val_results, val_gt, cfg)

            print(f"\n{'Threshold':>9} | {'TP':>4} | {'FP':>4} | {'Prec':>6} | "
                  f"{'Recall':>6} | {'F1':>6} | {'FP/case':>7}")
            print("-" * 55)

            best_f1_entry = None
            for entry in curves:
                print(f"{entry['threshold']:9.3f} | {entry['TP']:4d} | {entry['FP']:4d} | "
                      f"{entry['precision']:6.4f} | {entry['recall']:6.4f} | "
                      f"{entry['f1']:6.4f} | {entry['fp_per_case']:7.2f}")
                if best_f1_entry is None or entry["f1"] > best_f1_entry["f1"]:
                    best_f1_entry = entry

            print(f"\nBest F1: {best_f1_entry['f1']:.4f} at threshold={best_f1_entry['threshold']}")

            # Save curves
            with open(output_dir / "threshold_tuning.json", "w") as f:
                json.dump(curves, f, indent=2)

    # Save summary
    summary = {
        "threshold": threshold,
        "n_cases": len(case_ids),
        "per_case": {},
    }
    total_input = 0
    total_output = 0
    for case_id in case_ids:
        r = all_results[case_id]
        final_boxes, _ = threshold_filter(r, threshold)
        n_in = r["n_input"]
        n_out = len(final_boxes)
        total_input += n_in
        total_output += n_out
        summary["per_case"][case_id] = {"input": n_in, "output": n_out}

    summary["total_input"] = total_input
    summary["total_output"] = total_output
    summary["reduction_rate"] = 1 - total_output / total_input if total_input > 0 else 0

    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Total input predictions:  {total_input}")
    print(f"  Total output predictions: {total_output}")
    print(f"  Reduction rate: {summary['reduction_rate']:.1%}")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

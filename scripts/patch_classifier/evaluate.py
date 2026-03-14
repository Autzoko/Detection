"""
Evaluate 3D patch classifier post-processing on nnDetection predictions.

Reports (before and after classifier filtering):
  - Precision, recall, F1 at the operating threshold
  - FROC curve (sensitivity vs avg FPs/volume at FP rates 0.125..8)
  - Duplicate resolution rate
  - Per-volume breakdown flagging volumes where all GT lesions were filtered out

Usage:
    python evaluate.py --config config.yaml
    python evaluate.py --config config.yaml --threshold 0.5
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml
from skimage.measure import regionprops


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


def extract_gt_boxes(label_path):
    """Extract GT bboxes from instance segmentation NIfTI."""
    label_sitk = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_sitk).astype(np.int32)
    if label_arr.max() == 0:
        return []
    gt_boxes = []
    for prop in regionprops(label_arr):
        bbox = prop.bbox
        gt_boxes.append([bbox[0], bbox[1], bbox[3], bbox[4], bbox[2], bbox[5]])
    return gt_boxes


def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """Match predictions to GT boxes.

    Returns:
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives (unmatched GT)
        matches: list of (pred_idx, gt_idx, iou) for TPs
        gt_match_counts: dict mapping gt_idx -> number of predictions that matched it
    """
    gt_matched = [False] * len(gt_boxes)
    matches = []

    # Sort predictions by score descending
    sorted_indices = np.argsort(-pred_scores) if len(pred_scores) > 0 else []

    tp = 0
    fp = 0

    for p_idx in sorted_indices:
        best_iou = 0
        best_g = -1
        for g_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[g_idx]:
                continue
            ov = iou_3d(pred_boxes[p_idx].tolist(), gt_box)
            if ov > best_iou:
                best_iou = ov
                best_g = g_idx

        if best_iou >= iou_threshold and best_g >= 0:
            tp += 1
            gt_matched[best_g] = True
            matches.append((int(p_idx), best_g, best_iou))
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)

    # Count how many predictions overlap each GT (before matching)
    gt_match_counts = {}
    for g_idx, gt_box in enumerate(gt_boxes):
        count = 0
        for p_idx in range(len(pred_boxes)):
            if iou_3d(pred_boxes[p_idx].tolist(), gt_box) >= iou_threshold:
                count += 1
        gt_match_counts[g_idx] = count

    return tp, fp, fn, matches, gt_match_counts


def compute_froc(all_pred_boxes, all_pred_scores, all_gt_boxes,
                 iou_threshold, fp_rates):
    """Compute FROC curve.

    Sweep confidence thresholds and compute sensitivity at each specified
    average FP rate per volume.

    Returns dict mapping fp_rate -> sensitivity.
    """
    # Collect all predictions with case info
    all_preds = []
    total_gt = 0
    n_cases = len(all_gt_boxes)

    for case_id in all_gt_boxes:
        gt = all_gt_boxes[case_id]
        total_gt += len(gt)

        boxes = all_pred_boxes.get(case_id, np.zeros((0, 6)))
        scores = all_pred_scores.get(case_id, np.zeros(0))

        for i in range(len(boxes)):
            all_preds.append((float(scores[i]), case_id, i))

    if total_gt == 0:
        return {fp_rate: 0.0 for fp_rate in fp_rates}

    # Sort by score descending
    all_preds.sort(key=lambda x: -x[0])

    # Sweep through predictions
    sensitivities_at_fp = {}
    gt_found = set()  # (case_id, gt_idx)
    fp_count = 0

    pred_idx = 0
    for target_fp_rate in sorted(fp_rates):
        target_total_fp = target_fp_rate * n_cases

        while pred_idx < len(all_preds) and fp_count <= target_total_fp:
            score, case_id, box_idx = all_preds[pred_idx]
            gt = all_gt_boxes[case_id]
            boxes = all_pred_boxes[case_id]

            # Check if this prediction matches any unmatched GT
            matched = False
            best_iou = 0
            best_g = -1
            for g_idx, gt_box in enumerate(gt):
                if (case_id, g_idx) in gt_found:
                    continue
                ov = iou_3d(boxes[box_idx].tolist(), gt_box)
                if ov > best_iou:
                    best_iou = ov
                    best_g = g_idx

            if best_iou >= iou_threshold and best_g >= 0:
                gt_found.add((case_id, best_g))
            else:
                fp_count += 1

            pred_idx += 1

            if fp_count > target_total_fp:
                break

        sensitivity = len(gt_found) / total_gt
        sensitivities_at_fp[target_fp_rate] = round(sensitivity, 4)

    return sensitivities_at_fp


def compute_duplicate_resolution(before_counts, after_counts):
    """Compute duplicate resolution rate.

    before_counts: dict of gt_idx -> number of predictions matching it (before filtering)
    after_counts: dict of gt_idx -> number of predictions matching it (after filtering)

    Returns:
        n_dup_before: number of GT with >1 matching prediction before
        n_dup_resolved: number of those reduced to exactly 1
        resolution_rate: n_dup_resolved / n_dup_before
    """
    n_dup_before = sum(1 for c in before_counts.values() if c > 1)
    n_dup_resolved = 0

    for g_idx in before_counts:
        if before_counts[g_idx] > 1:
            after = after_counts.get(g_idx, 0)
            if after == 1:
                n_dup_resolved += 1

    rate = n_dup_resolved / n_dup_before if n_dup_before > 0 else 1.0
    return n_dup_before, n_dup_resolved, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--val_only", action="store_true",
                        help="Evaluate only on classifier val set")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    eval_cfg = cfg["evaluation"]
    iou_match = cfg["inference"]["iou_match_threshold"]
    threshold = args.threshold or cfg["inference"]["operating_threshold"]

    pred_dir = Path(paths["test_predictions_dir"])
    labels_dir = Path(paths["test_labels_dir"])
    output_dir = Path(paths["output_dir"])
    dataset_dir = Path(paths["dataset_dir"])

    # Load split info
    split_info_path = dataset_dir / "split_info.json"
    if split_info_path.exists():
        with open(split_info_path) as f:
            split_info = json.load(f)
        val_cases = set(split_info["val_cases"])
        train_cases = set(split_info["train_cases"])
    else:
        val_cases = set()
        train_cases = set()

    # Find all cases
    pred_files = sorted(pred_dir.glob("*_boxes.pkl"))
    all_case_ids = [f.stem.replace("_boxes", "") for f in pred_files]

    if args.val_only:
        case_ids = [c for c in all_case_ids if c in val_cases]
        print(f"Evaluating on val set only: {len(case_ids)} cases")
    else:
        case_ids = all_case_ids
        print(f"Evaluating on all {len(case_ids)} cases")

    # Load data
    all_gt = {}
    before_boxes = {}  # Raw nnDetection predictions
    before_scores = {}
    after_boxes = {}   # After classifier filtering
    after_scores = {}

    for case_id in case_ids:
        # GT
        gt_path = labels_dir / f"{case_id}.nii.gz"
        if gt_path.exists():
            all_gt[case_id] = extract_gt_boxes(gt_path)
        else:
            all_gt[case_id] = []

        # Before: raw nnDetection predictions
        with open(pred_dir / f"{case_id}_boxes.pkl", "rb") as f:
            raw_pred = pickle.load(f)
        before_boxes[case_id] = raw_pred["pred_boxes"]
        before_scores[case_id] = raw_pred["pred_scores"]

        # After: classifier-filtered predictions
        after_path = output_dir / f"{case_id}_boxes.pkl"
        if after_path.exists():
            with open(after_path, "rb") as f:
                filt_pred = pickle.load(f)
            after_boxes[case_id] = filt_pred["pred_boxes"]
            after_scores[case_id] = filt_pred["pred_scores"]
        else:
            after_boxes[case_id] = np.zeros((0, 6))
            after_scores[case_id] = np.zeros(0)

    total_gt = sum(len(g) for g in all_gt.values())
    print(f"Total GT lesions: {total_gt}")

    # ============================
    # Precision / Recall / F1
    # ============================
    print("\n" + "=" * 70)
    print(f"=== Precision / Recall / F1 (IoU >= {iou_match}) ===")
    print("=" * 70)

    for stage_name, boxes_dict, scores_dict in [
        ("BEFORE (raw nnDetection)", before_boxes, before_scores),
        (f"AFTER  (classifier, t={threshold})", after_boxes, after_scores),
    ]:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for case_id in case_ids:
            gt = all_gt[case_id]
            boxes = boxes_dict[case_id]
            scores = scores_dict[case_id]

            # Apply score threshold for "before" to make fair comparison
            if "BEFORE" in stage_name:
                mask = scores >= 0.3  # nnDetection default threshold
                boxes = boxes[mask]
                scores = scores[mask]

            tp, fp, fn, _, _ = match_predictions_to_gt(boxes, scores, gt, iou_match)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        n_cases = len(case_ids)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fp_per_case = total_fp / n_cases if n_cases > 0 else 0

        print(f"\n  {stage_name}:")
        print(f"    TP={total_tp}, FP={total_fp}, FN={total_fn}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    FP/case:   {fp_per_case:.2f}")

    # ============================
    # FROC Curve
    # ============================
    print("\n" + "=" * 70)
    print("=== FROC Curve ===")
    print("=" * 70)

    fp_rates = eval_cfg["froc_fp_rates"]

    for stage_name, boxes_dict, scores_dict in [
        ("BEFORE (raw nnDetection)", before_boxes, before_scores),
        ("AFTER  (classifier)", after_boxes, after_scores),
    ]:
        froc = compute_froc(boxes_dict, scores_dict, all_gt, iou_match, fp_rates)
        print(f"\n  {stage_name}:")
        print(f"    {'FP/case':>8} | {'Sensitivity':>11}")
        print(f"    {'-'*22}")
        for fp_rate in fp_rates:
            print(f"    {fp_rate:8.3f} | {froc[fp_rate]:11.4f}")

        mean_froc = np.mean(list(froc.values()))
        print(f"    Mean FROC: {mean_froc:.4f}")

    # ============================
    # Duplicate Resolution
    # ============================
    print("\n" + "=" * 70)
    print("=== Duplicate Resolution ===")
    print("=" * 70)

    total_dup_before = 0
    total_dup_resolved = 0

    for case_id in case_ids:
        gt = all_gt[case_id]
        if not gt:
            continue

        # Count matches before
        before_b = before_boxes[case_id]
        before_s = before_scores[case_id]
        mask = before_s >= 0.3
        before_b_f = before_b[mask]

        before_counts = {}
        for g_idx, gt_box in enumerate(gt):
            count = sum(1 for b in before_b_f if iou_3d(b.tolist(), gt_box) >= iou_match)
            before_counts[g_idx] = count

        # Count matches after
        after_b = after_boxes[case_id]
        after_counts = {}
        for g_idx, gt_box in enumerate(gt):
            count = sum(1 for b in after_b if iou_3d(b.tolist(), gt_box) >= iou_match)
            after_counts[g_idx] = count

        n_dup, n_resolved, _ = compute_duplicate_resolution(before_counts, after_counts)
        total_dup_before += n_dup
        total_dup_resolved += n_resolved

    dup_rate = total_dup_resolved / total_dup_before if total_dup_before > 0 else 1.0
    print(f"\n  GT lesions with multiple predictions (before): {total_dup_before}")
    print(f"  Successfully reduced to single prediction:     {total_dup_resolved}")
    print(f"  Duplicate resolution rate:                     {dup_rate:.4f}")

    # ============================
    # Per-Volume Breakdown
    # ============================
    print("\n" + "=" * 70)
    print("=== Per-Volume Breakdown ===")
    print("=" * 70)

    flagged_volumes = []

    print(f"\n  {'Case ID':>12} | {'GT':>3} | {'Before':>6} | {'After':>5} | "
          f"{'TP_before':>9} | {'TP_after':>8} | {'Status'}")
    print(f"  {'-'*70}")

    for case_id in case_ids:
        gt = all_gt[case_id]
        n_gt = len(gt)

        bb = before_boxes[case_id]
        bs = before_scores[case_id]
        mask = bs >= 0.3
        n_before = mask.sum()

        ab = after_boxes[case_id]
        a_s = after_scores[case_id]
        n_after = len(ab)

        # TP before
        tp_before, _, _, _, _ = match_predictions_to_gt(
            bb[mask], bs[mask], gt, iou_match) if n_gt > 0 else (0, 0, 0, [], {})
        # TP after
        tp_after, _, _, _, _ = match_predictions_to_gt(
            ab, a_s, gt, iou_match) if n_gt > 0 else (0, 0, 0, [], {})

        status = "OK"
        if n_gt > 0 and tp_after == 0:
            status = "ALL GT FILTERED OUT"
            flagged_volumes.append(case_id)
        elif tp_after < tp_before:
            status = f"LOST {tp_before - tp_after} TP"

        print(f"  {case_id:>12} | {n_gt:3d} | {n_before:6d} | {n_after:5d} | "
              f"{tp_before:9d} | {tp_after:8d} | {status}")

    if flagged_volumes:
        print(f"\n  WARNING: {len(flagged_volumes)} volumes had ALL GT lesions filtered out:")
        for v in flagged_volumes:
            print(f"    - {v}")
    else:
        print(f"\n  No volumes had all GT lesions filtered out.")

    # Save evaluation results
    eval_results = {
        "threshold": threshold,
        "iou_match": iou_match,
        "n_cases": len(case_ids),
        "total_gt": total_gt,
        "flagged_volumes": flagged_volumes,
        "duplicate_resolution_rate": dup_rate,
    }
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()

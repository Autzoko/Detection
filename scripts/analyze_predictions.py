"""
Comprehensive analysis of nnDetection predictions vs GT annotations.

Outputs:
  - per_case_metrics.csv: one row per case with all metrics
  - summary_stats.csv: aggregate statistics table
  - threshold_sweep.csv: precision/recall/F1 at each confidence threshold
  - report.txt: text summary of main findings
"""

import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# Config
# ============================================================
PRED_DIR = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/test_predictions"
RAW_DIR = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted"
LABELS_DIR = os.path.join(RAW_DIR, "labelsTs")
OUTPUT_DIR = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/prediction_analysis"
SPACING = np.array([1.0, 3.0, 1.0])  # z, y, x spacing in mm
DEFAULT_SCORE_THRESH = 0.3
NEAR_MISS_MM = 20.0


# ============================================================
# Helpers
# ============================================================
def extract_gt_boxes(label_path, json_path=None):
    """Extract GT bboxes from instance segmentation NIfTI.
    Returns list of dicts with 'box' [z_min,y_min,z_max,y_max,x_min,x_max] and 'class'.
    """
    label_sitk = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(label_sitk)  # (z, y, x)

    # Load class info from JSON if available
    instance_classes = {}
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        if "instances" in meta:
            instance_classes = {int(k): int(v) for k, v in meta["instances"].items()}

    gt_list = []
    for label_val in sorted(v for v in np.unique(arr) if v > 0):
        coords = np.argwhere(arr == label_val)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        box = [mins[0], mins[1], maxs[0], maxs[1], mins[2], maxs[2]]
        cls = instance_classes.get(int(label_val), 0)
        gt_list.append({"box": box, "class": cls, "instance_id": int(label_val)})
    return gt_list


def box_center(box):
    """Center of box [z_min,y_min,z_max,y_max,x_min,x_max] in voxels."""
    return np.array([
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2,
        (box[4] + box[5]) / 2,
    ])


def box_size(box):
    """Size of box per axis [dz, dy, dx]."""
    return np.array([box[2] - box[0], box[3] - box[1], box[5] - box[4]])


def center_distance_mm(box1, box2, spacing):
    """Euclidean distance between box centers in mm."""
    c1 = box_center(box1) * spacing
    c2 = box_center(box2) * spacing
    return np.linalg.norm(c1 - c2)


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


# ============================================================
# Main analysis
# ============================================================
def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_files = sorted(Path(PRED_DIR).glob("*_boxes.pkl"))
    case_ids = [f.stem.replace("_boxes", "") for f in pred_files]
    print(f"Cases: {len(case_ids)}")

    # ---- Per-case analysis ----
    per_case = []
    all_tp_scores = []
    all_fp_scores = []
    all_tp_ious = []
    all_fp_nearest_dist = []
    all_size_ratios = []  # (dz_ratio, dy_ratio, dx_ratio) for TP matches
    all_one_to_many_pairwise_iou = []
    all_one_to_many_pairwise_dist = []

    # For threshold sweep
    all_preds_with_gt = []  # (score, is_tp, case_id)
    total_gt_all = 0

    for case_id in case_ids:
        # Load predictions
        with open(os.path.join(PRED_DIR, f"{case_id}_boxes.pkl"), "rb") as f:
            pred = pickle.load(f)
        pred_boxes = pred["pred_boxes"]  # (N, 6)
        pred_scores = pred["pred_scores"]  # (N,)

        # Load GT
        gt_path = os.path.join(LABELS_DIR, f"{case_id}.nii.gz")
        json_path = os.path.join(LABELS_DIR, f"{case_id}.json")
        gt_list = extract_gt_boxes(gt_path, json_path)
        n_gt = len(gt_list)
        gt_boxes = [g["box"] for g in gt_list]
        total_gt_all += n_gt

        # Filter by default threshold
        mask = pred_scores >= DEFAULT_SCORE_THRESH
        filt_boxes = pred_boxes[mask]
        filt_scores = pred_scores[mask]
        n_pred = len(filt_boxes)

        # ---- Match preds to GT (greedy, by descending score) ----
        sorted_idx = np.argsort(-filt_scores)
        gt_matched = [False] * n_gt
        pred_is_tp = [False] * n_pred
        pred_match_gt = [-1] * n_pred
        pred_match_iou = [0.0] * n_pred

        for pi in sorted_idx:
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                ov = iou_3d(filt_boxes[pi], gb)
                if ov > best_iou:
                    best_iou = ov
                    best_gi = gi
            if best_iou >= 0.1 and best_gi >= 0:
                pred_is_tp[pi] = True
                pred_match_gt[pi] = best_gi
                pred_match_iou[pi] = best_iou
                gt_matched[best_gi] = True

        tp = sum(pred_is_tp)
        fp = n_pred - tp
        fn = sum(1 for m in gt_matched if not m)
        gt_hit = sum(gt_matched)
        gt_miss = n_gt - gt_hit

        # TP / FP score stats
        tp_scores = [float(filt_scores[i]) for i in range(n_pred) if pred_is_tp[i]]
        fp_scores_case = [float(filt_scores[i]) for i in range(n_pred) if not pred_is_tp[i]]
        all_tp_scores.extend(tp_scores)
        all_fp_scores.extend(fp_scores_case)

        tp_score_mean = np.mean(tp_scores) if tp_scores else 0
        tp_score_std = np.std(tp_scores) if tp_scores else 0
        fp_score_mean = np.mean(fp_scores_case) if fp_scores_case else 0
        fp_score_std = np.std(fp_scores_case) if fp_scores_case else 0

        # Average IoU of matched pairs
        matched_ious = [pred_match_iou[i] for i in range(n_pred) if pred_is_tp[i]]
        avg_iou = np.mean(matched_ious) if matched_ious else 0
        all_tp_ious.extend(matched_ious)

        # ---- One-to-many analysis (using ALL preds, not just greedy-matched) ----
        # For each GT, count how many predictions overlap it with IoU >= 0.1
        gt_overlapping_preds = defaultdict(list)
        for pi in range(n_pred):
            for gi, gb in enumerate(gt_boxes):
                if iou_3d(filt_boxes[pi], gb) >= 0.1:
                    gt_overlapping_preds[gi].append(pi)

        max_one_to_many = max((len(v) for v in gt_overlapping_preds.values()), default=0)

        # Pairwise stats for one-to-many groups
        for gi, pis in gt_overlapping_preds.items():
            if len(pis) > 1:
                for i in range(len(pis)):
                    for j in range(i + 1, len(pis)):
                        pw_iou = iou_3d(filt_boxes[pis[i]], filt_boxes[pis[j]])
                        pw_dist = center_distance_mm(
                            filt_boxes[pis[i]].tolist(), filt_boxes[pis[j]].tolist(), SPACING)
                        all_one_to_many_pairwise_iou.append(pw_iou)
                        all_one_to_many_pairwise_dist.append(pw_dist)

        # ---- Size ratio (TP pred vs GT) per axis ----
        for pi in range(n_pred):
            if pred_is_tp[pi]:
                gi = pred_match_gt[pi]
                ps = box_size(filt_boxes[pi].tolist())
                gs = box_size(gt_boxes[gi])
                ratio = ps / np.maximum(gs, 1e-6)
                all_size_ratios.append(ratio)

        # ---- FP analysis: distance to nearest GT ----
        fp_dists_case = []
        for pi in range(n_pred):
            if not pred_is_tp[pi]:
                if gt_boxes:
                    dists = [center_distance_mm(filt_boxes[pi].tolist(), gb, SPACING)
                             for gb in gt_boxes]
                    fp_dists_case.append(min(dists))
                else:
                    fp_dists_case.append(999.0)
        all_fp_nearest_dist.extend(fp_dists_case)
        n_near_miss = sum(1 for d in fp_dists_case if d <= NEAR_MISS_MM)
        n_isolated_fp = sum(1 for d in fp_dists_case if d > NEAR_MISS_MM)

        # ---- For threshold sweep (use ALL predictions, not just filtered) ----
        all_pred_boxes = pred_boxes
        all_pred_scores = pred_scores
        gt_matched_sweep = [False] * n_gt
        sweep_sorted = np.argsort(-all_pred_scores)
        for pi in sweep_sorted:
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gt_matched_sweep[gi]:
                    continue
                ov = iou_3d(all_pred_boxes[pi], gb)
                if ov > best_iou:
                    best_iou = ov
                    best_gi = gi
            is_tp = best_iou >= 0.1 and best_gi >= 0
            if is_tp:
                gt_matched_sweep[best_gi] = True
            all_preds_with_gt.append((float(all_pred_scores[pi]), is_tp, case_id))

        # ---- Store per-case row ----
        per_case.append({
            "case_id": case_id,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_hit": gt_hit,
            "gt_miss": gt_miss,
            "tp": tp,
            "fp": fp,
            "max_one_to_many": max_one_to_many,
            "tp_score_mean": round(tp_score_mean, 4),
            "tp_score_std": round(tp_score_std, 4),
            "fp_score_mean": round(fp_score_mean, 4),
            "fp_score_std": round(fp_score_std, 4),
            "avg_matched_iou": round(avg_iou, 4),
            "n_near_miss_fp": n_near_miss,
            "n_isolated_fp": n_isolated_fp,
        })

    # ============================================================
    # Aggregate statistics
    # ============================================================
    n_cases = len(case_ids)
    total_tp = sum(c["tp"] for c in per_case)
    total_fp = sum(c["fp"] for c in per_case)
    total_fn = sum(c["gt_miss"] for c in per_case)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt_all if total_gt_all > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Score separability
    tp_median = np.median(all_tp_scores) if all_tp_scores else 0
    fp_above_tp_median = sum(1 for s in all_fp_scores if s >= tp_median) if all_tp_scores else 0
    fp_above_tp_median_frac = fp_above_tp_median / len(all_fp_scores) if all_fp_scores else 0

    # Size ratios
    size_ratios = np.array(all_size_ratios) if all_size_ratios else np.zeros((0, 3))

    # FP distances
    fp_dists = np.array(all_fp_nearest_dist) if all_fp_nearest_dist else np.zeros(0)
    total_near_miss = sum(c["n_near_miss_fp"] for c in per_case)
    total_isolated = sum(c["n_isolated_fp"] for c in per_case)

    # One-to-many
    cases_with_otm = sum(1 for c in per_case if c["max_one_to_many"] > 1)
    otm_pairwise_iou = np.array(all_one_to_many_pairwise_iou) if all_one_to_many_pairwise_iou else np.zeros(0)
    otm_pairwise_dist = np.array(all_one_to_many_pairwise_dist) if all_one_to_many_pairwise_dist else np.zeros(0)

    # ============================================================
    # Threshold sweep
    # ============================================================
    thresholds = np.arange(0.1, 0.95, 0.05)
    sweep_rows = []
    for t in thresholds:
        # Re-do matching at this threshold
        sweep_tp = 0
        sweep_fp = 0
        sweep_fn_per_case = {}

        for case_id in case_ids:
            case_preds = [(s, is_tp) for s, is_tp, cid in all_preds_with_gt if cid == case_id and s >= t]
            # This is an approximation; for exact results we'd re-match.
            # Use the pre-computed matches: TP if was matched and score >= t
            ct = sum(1 for s, is_tp in case_preds if is_tp)
            cf = sum(1 for s, is_tp in case_preds if not is_tp)
            sweep_tp += ct
            sweep_fp += cf

        # FN = total_gt - sweep_tp (approximate, since matching depends on threshold)
        sweep_fn = total_gt_all - sweep_tp
        sp = sweep_tp / (sweep_tp + sweep_fp) if (sweep_tp + sweep_fp) > 0 else 0
        sr = sweep_tp / total_gt_all if total_gt_all > 0 else 0
        sf1 = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0
        fp_per_case = sweep_fp / n_cases

        sweep_rows.append({
            "threshold": round(float(t), 2),
            "TP": sweep_tp,
            "FP": sweep_fp,
            "FN": sweep_fn,
            "precision": round(sp, 4),
            "recall": round(sr, 4),
            "f1": round(sf1, 4),
            "fp_per_case": round(fp_per_case, 2),
        })

    # ============================================================
    # Write CSV: per_case_metrics.csv
    # ============================================================
    csv_path = os.path.join(OUTPUT_DIR, "per_case_metrics.csv")
    fields = list(per_case[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_case)
    print(f"Wrote {csv_path}")

    # ============================================================
    # Write CSV: threshold_sweep.csv
    # ============================================================
    sweep_path = os.path.join(OUTPUT_DIR, "threshold_sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"Wrote {sweep_path}")

    # ============================================================
    # Write CSV: summary_stats.csv
    # ============================================================
    summary = [
        ("Total cases", n_cases),
        ("Total GT lesions", total_gt_all),
        ("Score threshold", DEFAULT_SCORE_THRESH),
        ("", ""),
        ("--- Detection Performance ---", ""),
        ("Total TP", total_tp),
        ("Total FP", total_fp),
        ("Total FN (missed GT)", total_fn),
        ("Precision", round(precision, 4)),
        ("Recall", round(recall, 4)),
        ("F1", round(f1, 4)),
        ("Avg FP per case", round(total_fp / n_cases, 2)),
        ("", ""),
        ("--- TP Score Distribution ---", ""),
        ("TP count", len(all_tp_scores)),
        ("TP score mean", round(np.mean(all_tp_scores), 4) if all_tp_scores else "N/A"),
        ("TP score std", round(np.std(all_tp_scores), 4) if all_tp_scores else "N/A"),
        ("TP score median", round(np.median(all_tp_scores), 4) if all_tp_scores else "N/A"),
        ("TP score min", round(min(all_tp_scores), 4) if all_tp_scores else "N/A"),
        ("TP score max", round(max(all_tp_scores), 4) if all_tp_scores else "N/A"),
        ("", ""),
        ("--- FP Score Distribution ---", ""),
        ("FP count", len(all_fp_scores)),
        ("FP score mean", round(np.mean(all_fp_scores), 4) if all_fp_scores else "N/A"),
        ("FP score std", round(np.std(all_fp_scores), 4) if all_fp_scores else "N/A"),
        ("FP score median", round(np.median(all_fp_scores), 4) if all_fp_scores else "N/A"),
        ("FP above TP median", fp_above_tp_median),
        ("FP above TP median (%)", round(fp_above_tp_median_frac * 100, 1)),
        ("", ""),
        ("--- Matched IoU ---", ""),
        ("Avg matched IoU", round(np.mean(all_tp_ious), 4) if all_tp_ious else "N/A"),
        ("Median matched IoU", round(np.median(all_tp_ious), 4) if all_tp_ious else "N/A"),
        ("Min matched IoU", round(min(all_tp_ious), 4) if all_tp_ious else "N/A"),
        ("Max matched IoU", round(max(all_tp_ious), 4) if all_tp_ious else "N/A"),
        ("", ""),
        ("--- Pred/GT Size Ratio (TP only) ---", ""),
        ("Axis-Z (d0) ratio mean", round(float(size_ratios[:, 0].mean()), 3) if len(size_ratios) else "N/A"),
        ("Axis-Z (d0) ratio std", round(float(size_ratios[:, 0].std()), 3) if len(size_ratios) else "N/A"),
        ("Axis-Y (d1) ratio mean", round(float(size_ratios[:, 1].mean()), 3) if len(size_ratios) else "N/A"),
        ("Axis-Y (d1) ratio std", round(float(size_ratios[:, 1].std()), 3) if len(size_ratios) else "N/A"),
        ("Axis-X (d2) ratio mean", round(float(size_ratios[:, 2].mean()), 3) if len(size_ratios) else "N/A"),
        ("Axis-X (d2) ratio std", round(float(size_ratios[:, 2].std()), 3) if len(size_ratios) else "N/A"),
        ("", ""),
        ("--- One-to-Many ---", ""),
        ("Cases with one-to-many", cases_with_otm),
        ("Max one-to-many degree", max(c["max_one_to_many"] for c in per_case)),
        ("One-to-many pairwise IoU mean", round(float(otm_pairwise_iou.mean()), 4) if len(otm_pairwise_iou) else "N/A"),
        ("One-to-many pairwise dist mean (mm)", round(float(otm_pairwise_dist.mean()), 1) if len(otm_pairwise_dist) else "N/A"),
        ("", ""),
        ("--- FP Spatial Analysis ---", ""),
        ("Near-miss FPs (<=20mm from GT)", total_near_miss),
        ("Isolated FPs (>20mm from GT)", total_isolated),
        ("Near-miss fraction", round(total_near_miss / (total_near_miss + total_isolated), 3) if (total_near_miss + total_isolated) > 0 else "N/A"),
        ("FP nearest GT dist mean (mm)", round(float(fp_dists.mean()), 1) if len(fp_dists) else "N/A"),
        ("FP nearest GT dist median (mm)", round(float(np.median(fp_dists)), 1) if len(fp_dists) else "N/A"),
        ("FP nearest GT dist std (mm)", round(float(fp_dists.std()), 1) if len(fp_dists) else "N/A"),
    ]

    # Class-level breakdown
    class_counts = defaultdict(lambda: {"gt": 0, "tp": 0})
    for case_id in case_ids:
        gt_path = os.path.join(LABELS_DIR, f"{case_id}.nii.gz")
        json_path = os.path.join(LABELS_DIR, f"{case_id}.json")
        gt_list = extract_gt_boxes(gt_path, json_path)
        for g in gt_list:
            class_counts[g["class"]]["gt"] += 1

    if len(class_counts) > 1:
        summary.append(("", ""))
        summary.append(("--- Class-Level Breakdown ---", ""))
        for cls_id in sorted(class_counts.keys()):
            cc = class_counts[cls_id]
            summary.append((f"Class {cls_id}: GT count", cc["gt"]))

    summary_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerows(summary)
    print(f"Wrote {summary_path}")

    # ============================================================
    # Write report.txt
    # ============================================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("nnDetection Prediction Analysis Report")
    report_lines.append(f"Score threshold: {DEFAULT_SCORE_THRESH}")
    report_lines.append(f"Cases: {n_cases}, GT lesions: {total_gt_all}")
    report_lines.append("=" * 70)

    report_lines.append("")
    report_lines.append("1. DETECTION PERFORMANCE")
    report_lines.append(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    report_lines.append(f"   TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    report_lines.append(f"   Avg FP per case: {total_fp / n_cases:.1f}")

    report_lines.append("")
    report_lines.append("2. FALSE POSITIVE ANALYSIS")
    fp_med = round(total_fp / n_cases, 1)
    report_lines.append(f"   Typical FP count: {fp_med} per case (median: {np.median([c['fp'] for c in per_case]):.0f})")
    report_lines.append(f"   Near-miss FPs (within {NEAR_MISS_MM}mm of GT): {total_near_miss} ({total_near_miss/(total_near_miss+total_isolated)*100:.0f}%)" if (total_near_miss+total_isolated) > 0 else "   No FPs")
    report_lines.append(f"   Isolated FPs (>{NEAR_MISS_MM}mm from any GT): {total_isolated}")
    if len(fp_dists) > 0:
        report_lines.append(f"   FP-to-nearest-GT distance: mean={fp_dists.mean():.1f}mm, median={np.median(fp_dists):.1f}mm")

    report_lines.append("")
    report_lines.append("3. SCORE SEPARABILITY")
    if all_tp_scores and all_fp_scores:
        report_lines.append(f"   TP scores: mean={np.mean(all_tp_scores):.3f} ± {np.std(all_tp_scores):.3f} (median={np.median(all_tp_scores):.3f})")
        report_lines.append(f"   FP scores: mean={np.mean(all_fp_scores):.3f} ± {np.std(all_fp_scores):.3f} (median={np.median(all_fp_scores):.3f})")
        report_lines.append(f"   {fp_above_tp_median_frac*100:.1f}% of FPs score above the TP median ({tp_median:.3f})")
        if fp_above_tp_median_frac > 0.3:
            report_lines.append("   → POOR separability: FP and TP score distributions overlap heavily.")
            report_lines.append("     Score thresholding alone cannot cleanly separate TPs from FPs.")
        else:
            report_lines.append("   → GOOD separability: most FPs score below TP median.")
    else:
        report_lines.append("   Insufficient data for score analysis.")

    report_lines.append("")
    report_lines.append("4. ONE-TO-MANY DETECTIONS")
    report_lines.append(f"   Cases with one-to-many: {cases_with_otm}/{n_cases}")
    if cases_with_otm > 0:
        report_lines.append(f"   Max predictions per GT: {max(c['max_one_to_many'] for c in per_case)}")
        if len(otm_pairwise_iou) > 0:
            report_lines.append(f"   Pairwise IoU within groups: mean={otm_pairwise_iou.mean():.3f}")
            report_lines.append(f"   Pairwise center distance within groups: mean={otm_pairwise_dist.mean():.1f}mm")
        if cases_with_otm > n_cases * 0.3:
            report_lines.append("   → One-to-many is COMMON — duplicate resolution is important.")
        else:
            report_lines.append("   → One-to-many is moderate — some duplicate resolution needed.")

    report_lines.append("")
    report_lines.append("5. BOUNDING BOX SIZE ACCURACY")
    if len(size_ratios) > 0:
        report_lines.append(f"   Pred/GT size ratio (mean ± std):")
        report_lines.append(f"     Z-axis: {size_ratios[:,0].mean():.2f} ± {size_ratios[:,0].std():.2f}")
        report_lines.append(f"     Y-axis: {size_ratios[:,1].mean():.2f} ± {size_ratios[:,1].std():.2f}")
        report_lines.append(f"     X-axis: {size_ratios[:,2].mean():.2f} ± {size_ratios[:,2].std():.2f}")
        oversize = (size_ratios.mean(axis=1) > 1.2).sum()
        undersize = (size_ratios.mean(axis=1) < 0.8).sum()
        report_lines.append(f"   Oversized (>1.2x): {oversize}, Undersized (<0.8x): {undersize}")

    report_lines.append("")
    report_lines.append("6. MATCHED IoU QUALITY")
    if all_tp_ious:
        report_lines.append(f"   Mean IoU of matched pairs: {np.mean(all_tp_ious):.3f}")
        report_lines.append(f"   Median: {np.median(all_tp_ious):.3f}, Min: {min(all_tp_ious):.3f}, Max: {max(all_tp_ious):.3f}")
        low_iou = sum(1 for x in all_tp_ious if x < 0.2)
        report_lines.append(f"   Matches with IoU < 0.2 (marginal): {low_iou}/{len(all_tp_ious)}")

    report_lines.append("")
    report_lines.append("7. THRESHOLD SWEEP (best F1)")
    best_sweep = max(sweep_rows, key=lambda x: x["f1"])
    report_lines.append(f"   Best F1={best_sweep['f1']:.4f} at threshold={best_sweep['threshold']}")
    report_lines.append(f"   → P={best_sweep['precision']:.4f}, R={best_sweep['recall']:.4f}, FP/case={best_sweep['fp_per_case']:.1f}")

    report_lines.append("")
    report_lines.append("=" * 70)

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {report_path}")

    # Print report to console
    print("\n".join(report_lines))


if __name__ == "__main__":
    analyze()

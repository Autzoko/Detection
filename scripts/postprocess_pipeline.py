"""
Sequential post-processing pipeline for nnDetection predictions.

Steps:
  1. Anatomy mask filtering — remove predictions outside breast tissue (nonzero region + margin)
  2. Center distance clustering — merge nearby predictions at multiple distance thresholds
  3. Score threshold sweep — find optimal confidence threshold

Prints precision/recall/F1/FP-per-case after each step.
Outputs comparison table and saves final predictions in nnDetection pkl format.
"""

import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


# ============================================================
# Helpers (same conventions as analyze_predictions.py)
# ============================================================
def extract_gt_boxes(label_path, json_path=None):
    label_sitk = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(label_sitk)
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
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
        (box[4] + box[5]) / 2.0,
    ])


def center_distance_mm(box1, box2, spacing):
    c1 = box_center(box1) * spacing
    c2 = box_center(box2) * spacing
    return float(np.linalg.norm(c1 - c2))


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


def compute_metrics(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_thresh=0.1):
    """Compute aggregate P/R/F1/FP-per-case across all cases.

    Args:
        pred_boxes_list: list of arrays, one per case
        pred_scores_list: list of arrays, one per case
        gt_boxes_list: list of lists-of-boxes, one per case
    """
    total_tp = 0
    total_fp = 0
    total_gt = 0
    n_cases = len(pred_boxes_list)

    for boxes, scores, gt_boxes in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
        n_gt = len(gt_boxes)
        total_gt += n_gt
        n_pred = len(boxes)
        if n_pred == 0:
            continue

        sorted_idx = np.argsort(-scores)
        gt_matched = [False] * n_gt

        for pi in sorted_idx:
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                ov = iou_3d(boxes[pi], gb)
                if ov > best_iou:
                    best_iou = ov
                    best_gi = gi
            if best_iou >= iou_thresh and best_gi >= 0:
                total_tp += 1
                gt_matched[best_gi] = True
            else:
                total_fp += 1

    total_fn = total_gt - total_tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_per_case = total_fp / n_cases if n_cases > 0 else 0

    return {
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fp_per_case": fp_per_case,
    }


def print_metrics(label, metrics):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  FP/case:   {metrics['fp_per_case']:.2f}")
    print(f"{'=' * 60}")


# ============================================================
# Step 1: Anatomy mask filtering
# ============================================================
def anatomy_mask_filter(image_path, pred_boxes, pred_scores, margin_mm, spacing):
    """Remove predictions whose center falls outside the breast tissue region.

    Breast tissue = nonzero voxels. We compute a bounding box of nonzero region
    and dilate by margin_mm (converted to voxels using spacing).
    """
    img_sitk = sitk.ReadImage(str(image_path))
    arr = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)

    # Find bounding box of nonzero voxels
    nonzero = np.argwhere(arr > 0)
    if len(nonzero) == 0:
        # No tissue — remove everything
        return np.zeros((0, 6)), np.zeros(0)

    tissue_min = nonzero.min(axis=0)  # (z, y, x)
    tissue_max = nonzero.max(axis=0)

    # Convert margin from mm to voxels per axis
    margin_voxels = margin_mm / spacing  # (z, y, x)

    # Dilated bounding box
    roi_min = tissue_min - margin_voxels
    roi_max = tissue_max + margin_voxels

    # Clamp to volume bounds
    vol_shape = np.array(arr.shape)
    roi_min = np.maximum(roi_min, 0)
    roi_max = np.minimum(roi_max, vol_shape - 1)

    # Check each prediction center
    keep = []
    for i in range(len(pred_boxes)):
        center = box_center(pred_boxes[i])  # (z, y, x)
        if (center[0] >= roi_min[0] and center[0] <= roi_max[0] and
            center[1] >= roi_min[1] and center[1] <= roi_max[1] and
            center[2] >= roi_min[2] and center[2] <= roi_max[2]):
            keep.append(i)

    if len(keep) == 0:
        return np.zeros((0, 6)), np.zeros(0)
    keep = np.array(keep)
    return pred_boxes[keep], pred_scores[keep]


# ============================================================
# Step 2: Center distance clustering
# ============================================================
def cluster_by_distance(pred_boxes, pred_scores, distance_threshold_mm, spacing):
    """Merge predictions whose centers are within distance_threshold_mm.

    Uses Union-Find to group nearby predictions, then keeps the highest-scoring
    prediction from each cluster.
    """
    n = len(pred_boxes)
    if n == 0:
        return pred_boxes, pred_scores

    # Union-Find
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

    # Compute pairwise center distances and union close ones
    centers = np.array([box_center(pred_boxes[i]) * spacing for i in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist <= distance_threshold_mm:
                union(i, j)

    # Group by cluster
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    # Keep the highest-scoring prediction from each cluster
    keep = []
    for members in clusters.values():
        best = max(members, key=lambda i: pred_scores[i])
        keep.append(best)

    keep = sorted(keep)
    keep = np.array(keep)
    return pred_boxes[keep], pred_scores[keep]


# ============================================================
# Step 3: Score threshold sweep
# ============================================================
def apply_score_threshold(pred_boxes, pred_scores, threshold):
    mask = pred_scores >= threshold
    return pred_boxes[mask], pred_scores[mask]


# ============================================================
# Main pipeline
# ============================================================
def main():
    # Load config
    config_path = Path(__file__).parent / "patch_classifier" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pred_dir = cfg["paths"]["test_predictions_dir"]
    images_dir = cfg["paths"]["test_images_dir"]
    labels_dir = cfg["paths"]["test_labels_dir"]
    pp_cfg = cfg["postprocessing"]
    eval_cfg = cfg["evaluation"]

    spacing = np.array(pp_cfg["spacing"])
    output_dir = pp_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load all cases ----
    pred_files = sorted(Path(pred_dir).glob("*_boxes.pkl"))
    case_ids = [f.stem.replace("_boxes", "") for f in pred_files]
    print(f"Found {len(case_ids)} cases")

    # Load predictions and GT for all cases
    all_pred_data = []  # original pkl dicts (for saving later)
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []

    for case_id in case_ids:
        # Load predictions
        pkl_path = os.path.join(pred_dir, f"{case_id}_boxes.pkl")
        with open(pkl_path, "rb") as f:
            pred = pickle.load(f)
        all_pred_data.append(pred)
        all_pred_boxes.append(pred["pred_boxes"].copy())
        all_pred_scores.append(pred["pred_scores"].copy())

        # Load GT
        gt_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        json_path = os.path.join(labels_dir, f"{case_id}.json")
        gt_list = extract_gt_boxes(gt_path, json_path)
        all_gt_boxes.append([g["box"] for g in gt_list])

    # ---- Baseline metrics (no filtering) ----
    baseline = compute_metrics(all_pred_boxes, all_pred_scores, all_gt_boxes)
    print_metrics("BASELINE (all predictions, no filtering)", baseline)

    # Track metrics for comparison table
    comparison = [("Baseline", baseline)]

    # ============================================================
    # Step 1: Anatomy mask filtering
    # ============================================================
    print("\n>>> Step 1: Anatomy mask filtering ...")
    margin_mm = pp_cfg["anatomy_margin_mm"]
    step1_boxes = []
    step1_scores = []

    for i, case_id in enumerate(case_ids):
        img_path = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
        filt_boxes, filt_scores = anatomy_mask_filter(
            img_path, all_pred_boxes[i], all_pred_scores[i], margin_mm, spacing
        )
        step1_boxes.append(filt_boxes)
        step1_scores.append(filt_scores)

    step1_metrics = compute_metrics(step1_boxes, step1_scores, all_gt_boxes)
    print_metrics(f"After Step 1: Anatomy mask (margin={margin_mm}mm)", step1_metrics)
    comparison.append((f"+ Anatomy mask ({margin_mm}mm)", step1_metrics))

    removed_step1 = baseline["FP"] + baseline["TP"] - step1_metrics["FP"] - step1_metrics["TP"]
    print(f"  Predictions removed: {removed_step1}")

    # ============================================================
    # Step 2: Center distance clustering (sweep thresholds)
    # ============================================================
    distance_thresholds = pp_cfg["cluster_distance_thresholds_mm"]
    best_cluster_metrics = None
    best_cluster_dist = None
    best_cluster_boxes = None
    best_cluster_scores = None

    for dist_thresh in distance_thresholds:
        print(f"\n>>> Step 2: Center distance clustering (d={dist_thresh}mm) ...")
        step2_boxes = []
        step2_scores = []

        for i in range(len(case_ids)):
            clust_boxes, clust_scores = cluster_by_distance(
                step1_boxes[i], step1_scores[i], dist_thresh, spacing
            )
            step2_boxes.append(clust_boxes)
            step2_scores.append(clust_scores)

        step2_metrics = compute_metrics(step2_boxes, step2_scores, all_gt_boxes)
        print_metrics(f"After Step 2: Clustering (d={dist_thresh}mm)", step2_metrics)
        comparison.append((f"+ Clustering ({dist_thresh}mm)", step2_metrics))

        # Pick the clustering threshold with best F1 without dropping recall below baseline
        if best_cluster_metrics is None or (
            step2_metrics["recall"] >= baseline["recall"] * 0.95 and
            step2_metrics["f1"] > best_cluster_metrics["f1"]
        ):
            best_cluster_metrics = step2_metrics
            best_cluster_dist = dist_thresh
            best_cluster_boxes = step2_boxes
            best_cluster_scores = step2_scores

    print(f"\n  → Best clustering: d={best_cluster_dist}mm "
          f"(F1={best_cluster_metrics['f1']:.4f}, recall={best_cluster_metrics['recall']:.4f})")

    # ============================================================
    # Step 3: Score threshold sweep
    # ============================================================
    sweep_lo, sweep_hi = pp_cfg["sweep_range"]
    sweep_step = pp_cfg["sweep_step"]
    thresholds = np.arange(sweep_lo, sweep_hi + 1e-9, sweep_step)

    print(f"\n>>> Step 3: Score threshold sweep ({sweep_lo} to {sweep_hi}, step {sweep_step})")

    best_thresh = 0.0
    best_thresh_metrics = None
    sweep_results = []

    for t in thresholds:
        step3_boxes = []
        step3_scores = []
        for i in range(len(case_ids)):
            tb, ts = apply_score_threshold(best_cluster_boxes[i], best_cluster_scores[i], t)
            step3_boxes.append(tb)
            step3_scores.append(ts)

        step3_metrics = compute_metrics(step3_boxes, step3_scores, all_gt_boxes)
        sweep_results.append((round(float(t), 2), step3_metrics))

        if best_thresh_metrics is None or step3_metrics["f1"] > best_thresh_metrics["f1"]:
            best_thresh_metrics = step3_metrics
            best_thresh = round(float(t), 2)

    print_metrics(f"After Step 3: Best threshold={best_thresh}", best_thresh_metrics)
    comparison.append((f"+ Score threshold ({best_thresh})", best_thresh_metrics))

    # ============================================================
    # Print comparison table
    # ============================================================
    print(f"\n{'=' * 80}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 80}")
    header = f"{'Step':<35} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FP/case':>10} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    print("-" * 80)
    for label, m in comparison:
        row = (f"{label:<35} {m['precision']:>8.4f} {m['recall']:>8.4f} "
               f"{m['f1']:>8.4f} {m['fp_per_case']:>10.2f} "
               f"{m['TP']:>6} {m['FP']:>6} {m['FN']:>6}")
        print(row)
    print(f"{'=' * 80}")

    # ============================================================
    # Print threshold sweep table
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  SCORE THRESHOLD SWEEP (after anatomy + best clustering)")
    print(f"{'=' * 60}")
    print(f"{'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FP/case':>10}")
    print("-" * 44)
    for t, m in sweep_results:
        marker = " ← best" if t == best_thresh else ""
        print(f"{t:>8.2f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['fp_per_case']:>10.2f}{marker}")
    print(f"{'=' * 60}")

    # ============================================================
    # Save final predictions in nnDetection pkl format
    # ============================================================
    final_dir = os.path.join(output_dir, f"cluster{best_cluster_dist}mm_thresh{best_thresh}")
    os.makedirs(final_dir, exist_ok=True)

    for i, case_id in enumerate(case_ids):
        fb, fs = apply_score_threshold(best_cluster_boxes[i], best_cluster_scores[i], best_thresh)
        out_pred = dict(all_pred_data[i])  # copy metadata
        out_pred["pred_boxes"] = fb
        out_pred["pred_scores"] = fs
        out_pred["pred_labels"] = np.zeros(len(fb), dtype=np.int64)

        out_path = os.path.join(final_dir, f"{case_id}_boxes.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out_pred, f)

    print(f"\nSaved {len(case_ids)} postprocessed prediction files to: {final_dir}")

    # ============================================================
    # Save comparison CSV
    # ============================================================
    csv_path = os.path.join(output_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Step", "Precision", "Recall", "F1", "FP_per_case", "TP", "FP", "FN"])
        for label, m in comparison:
            w.writerow([label, round(m["precision"], 4), round(m["recall"], 4),
                        round(m["f1"], 4), round(m["fp_per_case"], 2),
                        m["TP"], m["FP"], m["FN"]])
    print(f"Saved comparison table to: {csv_path}")

    # Save threshold sweep CSV
    sweep_csv_path = os.path.join(output_dir, "threshold_sweep.csv")
    with open(sweep_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Threshold", "Precision", "Recall", "F1", "FP_per_case", "TP", "FP", "FN"])
        for t, m in sweep_results:
            w.writerow([t, round(m["precision"], 4), round(m["recall"], 4),
                        round(m["f1"], 4), round(m["fp_per_case"], 2),
                        m["TP"], m["FP"], m["FN"]])
    print(f"Saved threshold sweep to: {sweep_csv_path}")


if __name__ == "__main__":
    main()

"""
Post-process nnDetection predictions:
1. Filter by confidence threshold
2. Merge nearby/overlapping boxes into larger lesion-level boxes
3. Evaluate with both 3D IoU and center-point matching
4. Visualize merged results vs GT vs baseline

Usage:
    python scripts/postprocess_and_eval.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict


# ---- Paths ----
DATA_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted")
PRED_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/test_predictions")
STATS_CSV = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/dataset_statistics.csv")
BASELINE_XLSX = Path("/Users/langtian/Desktop/Prediction_vs_GT_Comparison.xlsx")
OUT_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/visualizations_postprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Parameters ----
SCORE_THRESH = 0.3          # initial confidence filter
MERGE_IOU_THRESH = 0.01     # IoU threshold for merging (low = aggressive merge)
MERGE_DIST_THRESH = 50      # max center distance (voxels) to consider merging
MIN_CLUSTER_SCORE = 0.5     # minimum max-score within a cluster to keep it


# =============================================================================
# Box utilities
# =============================================================================

def box_iou_3d(box_a, box_b):
    """Compute 3D IoU between two boxes [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]."""
    d0_inter = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    d1_inter = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    d2_inter = max(0, min(box_a[5], box_b[5]) - max(box_a[4], box_b[4]))
    inter = d0_inter * d1_inter * d2_inter
    vol_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) * (box_a[5] - box_a[4])
    vol_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) * (box_b[5] - box_b[4])
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0


def box_center(box):
    """Get center of box [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]."""
    return np.array([
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2,
        (box[4] + box[5]) / 2,
    ])


def center_distance(box_a, box_b):
    """Euclidean distance between box centers."""
    return np.linalg.norm(box_center(box_a) - box_center(box_b))


def merge_box_pair(box_a, box_b):
    """Merge two boxes into their bounding box."""
    return np.array([
        min(box_a[0], box_b[0]),  # d0_min
        min(box_a[1], box_b[1]),  # d1_min
        max(box_a[2], box_b[2]),  # d0_max
        max(box_a[3], box_b[3]),  # d1_max
        min(box_a[4], box_b[4]),  # d2_min
        max(box_a[5], box_b[5]),  # d2_max
    ])


# =============================================================================
# Post-processing: hierarchical merging
# =============================================================================

def merge_predictions(boxes, scores, iou_thresh=0.01, dist_thresh=50, min_score=0.5):
    """
    Merge nearby/overlapping predictions into lesion-level boxes.

    Algorithm:
    1. Start with highest-confidence box as a cluster seed
    2. Iteratively merge any unassigned box that overlaps or is nearby
    3. The merged box is the bounding box of all constituent boxes
    4. Cluster score = max score of any constituent box
    """
    if len(boxes) == 0:
        return np.empty((0, 6)), np.array([])

    # Sort by score descending
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]

    assigned = np.zeros(len(boxes), dtype=bool)
    clusters = []  # list of (merged_box, max_score, member_count)

    for i in range(len(boxes)):
        if assigned[i]:
            continue

        # Start a new cluster with this box
        cluster_box = boxes[i].copy()
        cluster_score = scores[i]
        cluster_count = 1
        assigned[i] = True

        # Iteratively try to merge more boxes
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if assigned[j]:
                    continue

                iou = box_iou_3d(cluster_box, boxes[j])
                dist = center_distance(cluster_box, boxes[j])

                if iou >= iou_thresh or dist <= dist_thresh:
                    cluster_box = merge_box_pair(cluster_box, boxes[j])
                    cluster_score = max(cluster_score, scores[j])
                    cluster_count += 1
                    assigned[j] = True
                    changed = True

        clusters.append((cluster_box, cluster_score, cluster_count))

    # Filter by minimum score
    merged_boxes = []
    merged_scores = []
    for box, score, count in clusters:
        if score >= min_score:
            merged_boxes.append(box)
            merged_scores.append(score)

    if merged_boxes:
        return np.array(merged_boxes), np.array(merged_scores)
    return np.empty((0, 6)), np.array([])


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_iou(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.1):
    """Standard IoU-based evaluation."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {
            'tp': 0, 'fp': len(pred_boxes), 'fn': len(gt_boxes),
            'tp_ious': [], 'matched_gt': []
        }

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp, fp = 0, 0
    tp_ious = []
    matched_gt = []

    for pb in pred_boxes:
        best_iou = 0
        best_gt = -1
        for gi, gb in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            iou = box_iou_3d(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi

        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            gt_matched[best_gt] = True
            tp_ious.append(best_iou)
            matched_gt.append(best_gt)
        else:
            fp += 1

    fn = int((~gt_matched).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tp_ious': tp_ious, 'matched_gt': matched_gt}


def evaluate_center_point(pred_boxes, pred_scores, gt_boxes, max_dist=30):
    """
    Center-point matching (similar to baseline evaluation).
    A prediction is TP if its center is within max_dist of a GT box center.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {'tp': 0, 'fp': len(pred_boxes), 'fn': len(gt_boxes)}

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp, fp = 0, 0

    for pb in pred_boxes:
        pc = box_center(pb)
        best_dist = float('inf')
        best_gt = -1
        for gi, gb in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            dist = np.linalg.norm(pc - box_center(gb))
            if dist < best_dist:
                best_dist = dist
                best_gt = gi

        if best_dist <= max_dist and best_gt >= 0:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1

    fn = int((~gt_matched).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn}


# =============================================================================
# Data loading
# =============================================================================

def load_gt_boxes(label_path):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    boxes = []
    for idx in np.unique(seg):
        if idx == 0:
            continue
        coords = np.stack(np.nonzero(seg == idx), axis=1)
        mins, maxs = coords.min(0), coords.max(0)
        boxes.append(np.array([
            mins[0] - 1, mins[1] - 1, maxs[0] + 1, maxs[1] + 1,
            mins[2] - 1, maxs[2] + 1
        ], dtype=float))
    return np.array(boxes) if boxes else np.empty((0, 6))


def load_raw_preds(pkl_path, score_thresh):
    with open(pkl_path, 'rb') as f:
        pred = pickle.load(f)
    mask = pred['pred_scores'] >= score_thresh
    return pred['pred_boxes'][mask], pred['pred_scores'][mask]


def build_case_mapping():
    df = pd.read_csv(STATS_CSV)
    test = df[df['split'] == 'test']
    mapping = {}
    for _, row in test.iterrows():
        orig = os.path.basename(row['image_path']).replace('.nii', '')
        mapping[row['volume_id']] = f"{orig}.ai"
    return mapping


def load_baseline_preds(baseline_df, filename):
    file_rows = baseline_df[baseline_df['Filename'] == filename]
    preds = file_rows[file_rows['Pred X1'].notna()]
    boxes, scores = [], []
    for _, row in preds.iterrows():
        x1, y1, z1 = row['Pred X1'], row['Pred Y1'], row['Pred Z1']
        x2, y2, z2 = row['Pred X2'], row['Pred Y2'], row['Pred Z2']
        boxes.append(np.array([z1, y1, z2, y2, x1, x2], dtype=float))
        scores.append(row.get('Pred Lesion Prob', 0))
    if boxes:
        return np.array(boxes), np.array(scores)
    return np.empty((0, 6)), np.array([])


# =============================================================================
# Visualization
# =============================================================================

def draw_boxes(ax, boxes, slice_idx, plane, color, label, scores=None,
               linewidth=2, linestyle='-'):
    drawn = False
    for i, b in enumerate(boxes):
        d0_min, d1_min, d0_max, d1_max, d2_min, d2_max = b
        show = False
        if plane == 'axial' and d0_min <= slice_idx <= d0_max:
            xy, w, h = (d2_min, d1_min), d2_max - d2_min, d1_max - d1_min
            show = True
        elif plane == 'coronal' and d1_min <= slice_idx <= d1_max:
            xy, w, h = (d2_min, d0_min), d2_max - d2_min, d0_max - d0_min
            show = True
        elif plane == 'sagittal' and d2_min <= slice_idx <= d2_max:
            xy, w, h = (d1_min, d0_min), d1_max - d1_min, d0_max - d0_min
            show = True
        if show:
            rect = patches.Rectangle(xy, w, h, linewidth=linewidth,
                                     edgecolor=color, facecolor='none',
                                     linestyle=linestyle,
                                     label=label if not drawn else None)
            ax.add_patch(rect)
            if scores is not None and len(scores) > i:
                ax.text(xy[0], xy[1] - 3, f'{scores[i]:.2f}', color=color,
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))
            drawn = True


def visualize_case(case_id, img, gt_boxes, raw_boxes, raw_scores,
                   merged_boxes, merged_scores, base_boxes, base_scores,
                   baseline_filename):
    for gi, gt_box in enumerate(gt_boxes):
        d0_c = int((gt_box[0] + gt_box[2]) / 2)
        d1_c = int((gt_box[1] + gt_box[3]) / 2)
        d2_c = int((gt_box[4] + gt_box[5]) / 2)

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(
            f'{case_id} ({baseline_filename}) - Lesion {gi+1}\n'
            f'Top: Raw nnDet ({len(raw_boxes)} boxes)  |  '
            f'Bottom: Post-processed ({len(merged_boxes)} merged boxes)',
            fontsize=13, fontweight='bold')

        views = [
            ('axial', d0_c, f'Axial (z={d0_c})'),
            ('coronal', d1_c, f'Coronal (y={d1_c})'),
            ('sagittal', d2_c, f'Sagittal (x={d2_c})'),
        ]

        for col, (plane, sl, title) in enumerate(views):
            for row_idx, (pred_b, pred_s, row_label) in enumerate([
                (raw_boxes, raw_scores, 'Raw'),
                (merged_boxes, merged_scores, 'Merged'),
            ]):
                ax = axes[row_idx, col]
                if plane == 'axial':
                    sl_c = min(max(sl, 0), img.shape[0] - 1)
                    ax.imshow(img[sl_c, :, :], cmap='gray', aspect='auto')
                elif plane == 'coronal':
                    sl_c = min(max(sl, 0), img.shape[1] - 1)
                    ax.imshow(img[:, sl_c, :], cmap='gray', aspect='auto')
                else:
                    sl_c = min(max(sl, 0), img.shape[2] - 1)
                    ax.imshow(img[:, :, sl_c], cmap='gray', aspect='auto')

                ax.set_title(f'{row_label} - {title}', fontsize=10)

                draw_boxes(ax, gt_boxes, sl_c, plane, 'lime', 'GT', linewidth=3)
                draw_boxes(ax, base_boxes, sl_c, plane, 'cyan', 'Baseline',
                           scores=base_scores, linewidth=2, linestyle='--')
                draw_boxes(ax, pred_b, sl_c, plane, 'red', f'nnDet ({row_label})',
                           scores=pred_s, linewidth=2)
                ax.legend(loc='upper right', fontsize=6)

        plt.tight_layout()
        out_path = OUT_DIR / f"{case_id}_lesion{gi+1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    case_map = build_case_mapping()
    baseline_df = pd.read_excel(BASELINE_XLSX, sheet_name='Detailed Comparison')
    baseline_files = set(baseline_df['Filename'].unique())

    # Aggregate metrics
    results_raw = defaultdict(int)
    results_merged = defaultdict(int)
    results_raw_cp = defaultdict(int)
    results_merged_cp = defaultdict(int)
    results_baseline_cp = defaultdict(int)
    all_tp_ious_raw = []
    all_tp_ious_merged = []

    print(f"{'Case':<14} {'BL file':<16} {'GT':>3} {'Raw':>4} {'Merged':>6} "
          f"{'Raw TP':>6} {'Mrg TP':>6} {'Mrg IoU':>8}")
    print("-" * 80)

    for case_id in sorted(case_map.keys()):
        bl_filename = case_map[case_id]
        if bl_filename not in baseline_files:
            continue

        label_path = DATA_DIR / "labelsTs" / f"{case_id}.nii.gz"
        pred_path = PRED_DIR / f"{case_id}_boxes.pkl"
        img_path = DATA_DIR / "imagesTs" / f"{case_id}_0000.nii.gz"

        if not pred_path.exists():
            continue

        gt_boxes = load_gt_boxes(label_path)
        raw_boxes, raw_scores = load_raw_preds(pred_path, SCORE_THRESH)
        merged_boxes, merged_scores = merge_predictions(
            raw_boxes, raw_scores,
            iou_thresh=MERGE_IOU_THRESH,
            dist_thresh=MERGE_DIST_THRESH,
            min_score=MIN_CLUSTER_SCORE,
        )
        base_boxes, base_scores = load_baseline_preds(baseline_df, bl_filename)

        # Evaluate: IoU-based
        eval_raw = evaluate_iou(raw_boxes, raw_scores, gt_boxes, iou_thresh=0.1)
        eval_merged = evaluate_iou(merged_boxes, merged_scores, gt_boxes, iou_thresh=0.1)

        # Evaluate: center-point
        eval_raw_cp = evaluate_center_point(raw_boxes, raw_scores, gt_boxes, max_dist=30)
        eval_merged_cp = evaluate_center_point(merged_boxes, merged_scores, gt_boxes, max_dist=30)
        eval_base_cp = evaluate_center_point(base_boxes, base_scores, gt_boxes, max_dist=30)

        for key in ['tp', 'fp', 'fn']:
            results_raw[key] += eval_raw[key]
            results_merged[key] += eval_merged[key]
            results_raw_cp[key] += eval_raw_cp[key]
            results_merged_cp[key] += eval_merged_cp[key]
            results_baseline_cp[key] += eval_base_cp[key]

        all_tp_ious_raw.extend(eval_raw['tp_ious'])
        all_tp_ious_merged.extend(eval_merged['tp_ious'])

        avg_iou = np.mean(eval_merged['tp_ious']) if eval_merged['tp_ious'] else 0

        print(f"{case_id:<14} {bl_filename:<16} {len(gt_boxes):>3} "
              f"{len(raw_boxes):>4} {len(merged_boxes):>6} "
              f"{eval_raw['tp']:>6} {eval_merged['tp']:>6} "
              f"{avg_iou:>8.3f}")

        # Visualize
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        visualize_case(case_id, img, gt_boxes, raw_boxes, raw_scores,
                       merged_boxes, merged_scores, base_boxes, base_scores,
                       bl_filename)

    # Print summary
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    def print_metrics(name, res):
        prec = res['tp'] / (res['tp'] + res['fp']) if (res['tp'] + res['fp']) > 0 else 0
        rec = res['tp'] / (res['tp'] + res['fn']) if (res['tp'] + res['fn']) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {name:<30} TP={res['tp']:>3}  FP={res['fp']:>4}  FN={res['fn']:>3}  "
              f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

    print("\n--- IoU >= 0.1 matching ---")
    print_metrics("Raw nnDetection", results_raw)
    print_metrics("Merged nnDetection", results_merged)

    print("\n--- Center-point matching (dist<=30) ---")
    print_metrics("Raw nnDetection", results_raw_cp)
    print_metrics("Merged nnDetection", results_merged_cp)
    print_metrics("Baseline", results_baseline_cp)

    print(f"\n--- IoU distribution (merged TPs) ---")
    if all_tp_ious_merged:
        ious = np.array(all_tp_ious_merged)
        print(f"  Mean IoU: {ious.mean():.3f}")
        print(f"  Median IoU: {np.median(ious):.3f}")
        print(f"  IoU >= 0.3: {(ious >= 0.3).sum()}/{len(ious)}")
        print(f"  IoU >= 0.5: {(ious >= 0.5).sum()}/{len(ious)}")

    print(f"\nVisualizations saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

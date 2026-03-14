"""
Post-processing module for nnDetection predictions on 3D breast ultrasound.

Pipeline:
  1. Confidence thresholding
  2. Size & aspect ratio filtering (based on dataset statistics)
  3. IoU + center-distance clustering → merge into lesion-level boxes
  4. Support counting (how many raw predictions back each cluster)
  5. NMS on merged boxes
  6. Evaluation & visualization

Box format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
  d0=z (500), d1=y (350), d2=x (1017)

Usage:
    python scripts/postprocess_fp_reduction.py
    python scripts/postprocess_fp_reduction.py --score_thresh 0.5 --visualize
"""

import os
import pickle
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PostProcessConfig:
    """All tunable parameters in one place."""
    # --- Paths ---
    data_dir: str = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted"
    pred_dir: str = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/test_predictions"
    out_dir: str = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/postprocessed_v2"

    # --- Stage 1: Confidence threshold ---
    score_thresh: float = 0.3

    # --- Stage 2: Size & aspect ratio filter ---
    # Minimum box span per axis (voxels) — remove tiny spurious boxes
    min_size_d0: int = 10
    min_size_d1: int = 5
    min_size_d2: int = 15
    # Maximum box span per axis — remove implausibly large boxes
    max_size_d0: int = 400
    max_size_d1: int = 200
    max_size_d2: int = 700
    # Aspect ratio: d2/d0 range (x/z)
    min_aspect_d2_d0: float = 0.3
    max_aspect_d2_d0: float = 10.0
    # Aspect ratio: d2/d1 range (x/y)
    min_aspect_d2_d1: float = 0.5
    max_aspect_d2_d1: float = 30.0

    # --- Stage 3: Clustering / merging ---
    # Two boxes are candidates for merging if IoU > merge_iou OR center dist < merge_dist
    merge_iou_thresh: float = 0.01
    merge_center_dist: float = 50.0

    # --- Stage 4: Support filter ---
    # Minimum number of raw predictions in a cluster to keep it
    min_support: int = 1

    # --- Stage 5: NMS on merged boxes ---
    nms_iou_thresh: float = 0.3

    # --- Evaluation ---
    eval_iou_thresh: float = 0.1
    eval_center_dist: float = 50.0

    # --- Visualization ---
    visualize: bool = False
    vis_dir: str = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/visualizations_pp_v2"
    num_vis_cases: Optional[int] = None  # None = all


# =============================================================================
# Box utilities
# =============================================================================

def box_volume(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1]) * max(0, b[5]-b[4])


def box_iou_3d(a, b):
    d0 = max(0, min(a[2],b[2]) - max(a[0],b[0]))
    d1 = max(0, min(a[3],b[3]) - max(a[1],b[1]))
    d2 = max(0, min(a[5],b[5]) - max(a[4],b[4]))
    inter = d0 * d1 * d2
    union = box_volume(a) + box_volume(b) - inter
    return inter / union if union > 0 else 0.0


def box_center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2, (b[4]+b[5])/2])


def box_spans(b):
    return b[2]-b[0], b[3]-b[1], b[5]-b[4]


def merge_box_cluster(boxes, scores):
    """Merge a cluster of boxes into one bounding box via outer union."""
    merged = np.array([
        boxes[:, 0].min(),
        boxes[:, 1].min(),
        boxes[:, 2].max(),
        boxes[:, 3].max(),
        boxes[:, 4].min(),
        boxes[:, 5].max(),
    ], dtype=float)
    # Score: max of members
    max_score = scores.max()
    mean_score = scores.mean()
    return merged, max_score, mean_score


# =============================================================================
# Pipeline stages
# =============================================================================

def stage1_score_filter(boxes, scores, thresh):
    mask = scores >= thresh
    return boxes[mask], scores[mask], mask.sum()


def stage2_size_aspect_filter(boxes, scores, cfg):
    keep = []
    for i, b in enumerate(boxes):
        s0, s1, s2 = box_spans(b)

        # Size filter
        if s0 < cfg.min_size_d0 or s1 < cfg.min_size_d1 or s2 < cfg.min_size_d2:
            continue
        if s0 > cfg.max_size_d0 or s1 > cfg.max_size_d1 or s2 > cfg.max_size_d2:
            continue

        # Aspect ratio filter
        if s0 > 0 and s2 > 0:
            ratio_d2_d0 = s2 / s0
            if ratio_d2_d0 < cfg.min_aspect_d2_d0 or ratio_d2_d0 > cfg.max_aspect_d2_d0:
                continue
        if s1 > 0 and s2 > 0:
            ratio_d2_d1 = s2 / s1
            if ratio_d2_d1 < cfg.min_aspect_d2_d1 or ratio_d2_d1 > cfg.max_aspect_d2_d1:
                continue

        keep.append(i)

    keep = np.array(keep, dtype=int)
    if len(keep) == 0:
        return np.empty((0, 6)), np.array([]), 0
    return boxes[keep], scores[keep], len(boxes) - len(keep)


def stage3_cluster_merge(boxes, scores, cfg):
    """
    Greedy clustering: seed from highest-confidence box, iteratively add
    any box with IoU > thresh OR center distance < thresh to the cluster.
    Then merge each cluster into a single outer bounding box.
    """
    n = len(boxes)
    if n == 0:
        return np.empty((0, 6)), np.array([]), np.array([]), np.array([]), []

    used = np.zeros(n, dtype=bool)
    order = np.argsort(-scores)

    merged_boxes = []
    merged_max_scores = []
    merged_mean_scores = []
    merged_support = []
    cluster_members = []

    for seed in order:
        if used[seed]:
            continue
        used[seed] = True
        cluster = [seed]

        # Iteratively expand
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if used[j]:
                    continue
                # Check against all current cluster members
                for k in cluster:
                    iou = box_iou_3d(boxes[k], boxes[j])
                    dist = np.linalg.norm(box_center(boxes[k]) - box_center(boxes[j]))
                    if iou > cfg.merge_iou_thresh or dist < cfg.merge_center_dist:
                        cluster.append(j)
                        used[j] = True
                        changed = True
                        break

        cl_boxes = boxes[cluster]
        cl_scores = scores[cluster]
        merged, max_s, mean_s = merge_box_cluster(cl_boxes, cl_scores)

        merged_boxes.append(merged)
        merged_max_scores.append(max_s)
        merged_mean_scores.append(mean_s)
        merged_support.append(len(cluster))
        cluster_members.append(cluster)

    return (np.array(merged_boxes), np.array(merged_max_scores),
            np.array(merged_mean_scores), np.array(merged_support),
            cluster_members)


def stage4_support_filter(boxes, scores, mean_scores, support, members, cfg):
    """Remove clusters with too few supporting raw predictions."""
    mask = support >= cfg.min_support
    kept_members = [m for m, k in zip(members, mask) if k]
    return (boxes[mask], scores[mask], mean_scores[mask],
            support[mask], kept_members, (~mask).sum())


def stage5_nms(boxes, scores, cfg):
    """Standard 3D NMS on merged boxes."""
    if len(boxes) == 0:
        return np.empty((0, 6)), np.array([]), []

    order = np.argsort(-scores)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j in suppressed or j == i:
                continue
            if box_iou_3d(boxes[i], boxes[j]) > cfg.nms_iou_thresh:
                suppressed.add(j)

    keep = np.array(keep)
    return boxes[keep], scores[keep], keep.tolist()


# =============================================================================
# GT loading & evaluation
# =============================================================================

def load_gt_boxes(label_path):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    boxes = []
    for idx in np.unique(seg)[1:]:
        coords = np.stack(np.nonzero(seg == idx), axis=1)
        mins, maxs = coords.min(0), coords.max(0)
        boxes.append(np.array([mins[0]-1, mins[1]-1, maxs[0]+1, maxs[1]+1,
                                mins[2]-1, maxs[2]+1], dtype=float))
    return np.array(boxes) if boxes else np.empty((0, 6))


def evaluate_case(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.1):
    """Per-case evaluation: greedy matching by score, IoU threshold."""
    matched_gt = set()
    tp_ious = []

    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        order = np.argsort(-pred_scores)
        for pi in order:
            best_iou, best_gi = 0, -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                iou = box_iou_3d(pred_boxes[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thresh and best_gi >= 0:
                matched_gt.add(best_gi)
                tp_ious.append(best_iou)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, tp_ious


def evaluate_case_center(pred_boxes, pred_scores, gt_boxes, dist_thresh=50):
    """Center-point matching evaluation."""
    matched_gt = set()
    tp = 0

    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        order = np.argsort(-pred_scores)
        for pi in order:
            pc = box_center(pred_boxes[pi])
            best_dist, best_gi = float('inf'), -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                gc = box_center(gt_boxes[gi])
                dist = np.linalg.norm(pc - gc)
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi
            if best_dist < dist_thresh and best_gi >= 0:
                matched_gt.add(best_gi)
                tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# =============================================================================
# Visualization
# =============================================================================

def visualize_case(case_id, img, gt_boxes, raw_boxes, raw_scores,
                   pp_boxes, pp_scores, pp_support, vis_dir, cfg):
    """Three-row comparison: raw vs post-processed vs GT for each GT lesion."""
    if len(gt_boxes) == 0:
        return

    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for gi, gt_box in enumerate(gt_boxes):
        d0_c = int((gt_box[0] + gt_box[2]) / 2)
        d1_c = int((gt_box[1] + gt_box[3]) / 2)
        d2_c = int((gt_box[4] + gt_box[5]) / 2)

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(
            f'{case_id} - Lesion {gi+1}\n'
            f'Raw: {len(raw_boxes)} boxes (s>={cfg.score_thresh})  |  '
            f'Post-processed: {len(pp_boxes)} boxes\n'
            f'Green=GT  |  Red=Raw  |  Blue=Post-processed',
            fontsize=13, fontweight='bold')

        views = [
            ('axial', d0_c, f'Axial (z={d0_c})'),
            ('coronal', d1_c, f'Coronal (y={d1_c})'),
            ('sagittal', d2_c, f'Sagittal (x={d2_c})'),
        ]

        for col, (plane, sl, title) in enumerate(views):
            # Row 0: Raw predictions
            ax = axes[0, col]
            _show_slice(ax, img, plane, sl)
            ax.set_title(f'RAW - {title}', fontsize=10)
            _draw_boxes(ax, gt_boxes, sl, plane, 'lime', 'GT', linewidth=3)
            _draw_boxes(ax, raw_boxes, sl, plane, 'red', 'Raw',
                        scores=raw_scores, linewidth=1, linestyle='-', alpha=0.5)

            # Row 1: Post-processed
            ax = axes[1, col]
            _show_slice(ax, img, plane, sl)
            ax.set_title(f'POST-PROCESSED - {title}', fontsize=10)
            _draw_boxes(ax, gt_boxes, sl, plane, 'lime', 'GT', linewidth=3)
            _draw_boxes(ax, pp_boxes, sl, plane, 'cyan', 'PP',
                        scores=pp_scores, support=pp_support,
                        linewidth=2, linestyle='-')

            for a in [axes[0, col], axes[1, col]]:
                a.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        out_path = vis_dir / f"{case_id}_lesion{gi+1}_pp.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()


def _show_slice(ax, img, plane, sl):
    if plane == 'axial':
        sl = min(max(sl, 0), img.shape[0]-1)
        ax.imshow(img[sl, :, :], cmap='gray', aspect='auto')
    elif plane == 'coronal':
        sl = min(max(sl, 0), img.shape[1]-1)
        ax.imshow(img[:, sl, :], cmap='gray', aspect='auto')
    else:
        sl = min(max(sl, 0), img.shape[2]-1)
        ax.imshow(img[:, :, sl], cmap='gray', aspect='auto')


def _draw_boxes(ax, boxes, slice_idx, plane, color, label,
                scores=None, support=None, linewidth=2, linestyle='-', alpha=1.0):
    drawn = False
    for i, b in enumerate(boxes):
        d0_min, d1_min, d0_max, d1_max, d2_min, d2_max = b
        show = False
        if plane == 'axial' and d0_min <= slice_idx <= d0_max:
            xy = (d2_min, d1_min); w = d2_max-d2_min; h = d1_max-d1_min; show = True
        elif plane == 'coronal' and d1_min <= slice_idx <= d1_max:
            xy = (d2_min, d0_min); w = d2_max-d2_min; h = d0_max-d0_min; show = True
        elif plane == 'sagittal' and d2_min <= slice_idx <= d2_max:
            xy = (d1_min, d0_min); w = d1_max-d1_min; h = d0_max-d0_min; show = True

        if show:
            rect = mpatches.Rectangle(
                xy, w, h, linewidth=linewidth, edgecolor=color,
                facecolor='none', linestyle=linestyle, alpha=alpha,
                label=label if not drawn else None)
            ax.add_patch(rect)
            txt_parts = []
            if scores is not None and i < len(scores):
                txt_parts.append(f'{scores[i]:.2f}')
            if support is not None and i < len(support):
                txt_parts.append(f'n={support[i]}')
            if txt_parts:
                ax.text(xy[0], xy[1]-3, ' '.join(txt_parts), color=color,
                        fontsize=6, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))
            drawn = True


# =============================================================================
# Main pipeline
# =============================================================================

def process_case(case_id, cfg):
    """Run full post-processing pipeline on a single case."""
    pred_path = Path(cfg.pred_dir) / f"{case_id}_boxes.pkl"
    label_path = Path(cfg.data_dir) / "labelsTs" / f"{case_id}.nii.gz"

    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)

    raw_boxes = pred['pred_boxes']
    raw_scores = pred['pred_scores']
    gt_boxes = load_gt_boxes(label_path)

    stats = {'case_id': case_id, 'n_raw': len(raw_boxes), 'n_gt': len(gt_boxes)}

    # Stage 1: Score threshold
    boxes, scores, _ = stage1_score_filter(raw_boxes, raw_scores, cfg.score_thresh)
    stats['n_after_score'] = len(boxes)

    # Stage 2: Size & aspect ratio
    boxes, scores, n_removed = stage2_size_aspect_filter(boxes, scores, cfg)
    stats['n_after_size'] = len(boxes)
    stats['n_size_removed'] = n_removed

    # Stage 3: Cluster & merge
    (m_boxes, m_max_scores, m_mean_scores,
     m_support, m_members) = stage3_cluster_merge(boxes, scores, cfg)
    stats['n_clusters'] = len(m_boxes)

    # Stage 4: Support filter
    (m_boxes, m_max_scores, m_mean_scores,
     m_support, m_members, n_removed) = stage4_support_filter(
        m_boxes, m_max_scores, m_mean_scores, m_support, m_members, cfg)
    stats['n_after_support'] = len(m_boxes)
    stats['n_support_removed'] = n_removed

    # Stage 5: NMS
    m_boxes, m_max_scores, keep_idx = stage5_nms(m_boxes, m_max_scores, cfg)
    m_support = m_support[keep_idx] if len(keep_idx) > 0 else np.array([])
    m_mean_scores = m_mean_scores[keep_idx] if len(keep_idx) > 0 else np.array([])
    stats['n_final'] = len(m_boxes)

    # Evaluate raw (score-filtered only)
    raw_filt_boxes, raw_filt_scores, _ = stage1_score_filter(
        raw_boxes, raw_scores, cfg.score_thresh)
    tp_r, fp_r, fn_r, _ = evaluate_case(
        raw_filt_boxes, raw_filt_scores, gt_boxes, cfg.eval_iou_thresh)
    stats['raw_tp'], stats['raw_fp'], stats['raw_fn'] = tp_r, fp_r, fn_r

    # Evaluate post-processed
    tp_p, fp_p, fn_p, tp_ious = evaluate_case(
        m_boxes, m_max_scores, gt_boxes, cfg.eval_iou_thresh)
    stats['pp_tp'], stats['pp_fp'], stats['pp_fn'] = tp_p, fp_p, fn_p
    stats['pp_ious'] = tp_ious

    # Center-point eval
    tp_c, fp_c, fn_c = evaluate_case_center(
        m_boxes, m_max_scores, gt_boxes, cfg.eval_center_dist)
    stats['pp_center_tp'] = tp_c
    stats['pp_center_fp'] = fp_c
    stats['pp_center_fn'] = fn_c

    log.info(f"  {case_id}: raw={stats['n_raw']} -> score={stats['n_after_score']} "
             f"-> size={stats['n_after_size']} -> clusters={stats['n_clusters']} "
             f"-> support={stats['n_after_support']} -> NMS={stats['n_final']}  |  "
             f"GT={stats['n_gt']}  TP={tp_p} FP={fp_p} FN={fn_p}")

    # Save post-processed predictions
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pp_result = {
        'pred_boxes': m_boxes,
        'pred_scores': m_max_scores,
        'pred_mean_scores': m_mean_scores,
        'pred_support': m_support,
        'pred_labels': np.zeros(len(m_boxes), dtype=int),
        'original_size_of_raw_data': pred.get('original_size_of_raw_data'),
        'itk_origin': pred.get('itk_origin'),
        'itk_spacing': pred.get('itk_spacing'),
        'itk_direction': pred.get('itk_direction'),
    }
    with open(out_dir / f"{case_id}_boxes.pkl", 'wb') as f:
        pickle.dump(pp_result, f)

    # Visualization
    if cfg.visualize:
        img_path = Path(cfg.data_dir) / "imagesTs" / f"{case_id}_0000.nii.gz"
        if img_path.exists():
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
            visualize_case(case_id, img, gt_boxes,
                           raw_filt_boxes, raw_filt_scores,
                           m_boxes, m_max_scores, m_support,
                           cfg.vis_dir, cfg)

    return stats


def run_pipeline(cfg):
    """Run on all test cases and produce summary."""
    pred_dir = Path(cfg.pred_dir)
    case_ids = sorted([p.stem.replace('_boxes', '')
                       for p in pred_dir.glob('*_boxes.pkl')])
    log.info(f"Found {len(case_ids)} cases")
    log.info(f"Config: score>={cfg.score_thresh}, min_support>={cfg.min_support}, "
             f"merge_dist={cfg.merge_center_dist}, nms_iou={cfg.nms_iou_thresh}")
    log.info("")

    all_stats = []
    for case_id in case_ids:
        stats = process_case(case_id, cfg)
        all_stats.append(stats)

    # ---- Aggregate results ----
    log.info("\n" + "=" * 70)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 70)

    total_raw = sum(s['n_raw'] for s in all_stats)
    total_after_score = sum(s['n_after_score'] for s in all_stats)
    total_after_size = sum(s['n_after_size'] for s in all_stats)
    total_clusters = sum(s['n_clusters'] for s in all_stats)
    total_final = sum(s['n_final'] for s in all_stats)
    total_gt = sum(s['n_gt'] for s in all_stats)

    log.info(f"\nBox count reduction:")
    log.info(f"  Raw predictions:    {total_raw:5d}")
    log.info(f"  After score filter: {total_after_score:5d}  (thresh={cfg.score_thresh})")
    log.info(f"  After size filter:  {total_after_size:5d}")
    log.info(f"  After clustering:   {total_clusters:5d}")
    log.info(f"  Final (NMS):        {total_final:5d}")
    log.info(f"  GT lesions:         {total_gt:5d}")
    n_cases = len(all_stats)

    # Raw evaluation
    raw_tp = sum(s['raw_tp'] for s in all_stats)
    raw_fp = sum(s['raw_fp'] for s in all_stats)
    raw_fn = sum(s['raw_fn'] for s in all_stats)
    raw_p = raw_tp / (raw_tp + raw_fp) if raw_tp + raw_fp > 0 else 0
    raw_r = raw_tp / total_gt if total_gt > 0 else 0
    raw_f1 = 2*raw_p*raw_r / (raw_p+raw_r) if raw_p+raw_r > 0 else 0

    log.info(f"\nRaw (score>={cfg.score_thresh} only):")
    log.info(f"  TP={raw_tp}, FP={raw_fp}, FN={raw_fn}")
    log.info(f"  Precision={raw_p:.4f}, Recall={raw_r:.4f}, F1={raw_f1:.4f}")
    log.info(f"  FP/volume={raw_fp/n_cases:.1f}")

    # Post-processed evaluation (IoU)
    pp_tp = sum(s['pp_tp'] for s in all_stats)
    pp_fp = sum(s['pp_fp'] for s in all_stats)
    pp_fn = sum(s['pp_fn'] for s in all_stats)
    pp_p = pp_tp / (pp_tp + pp_fp) if pp_tp + pp_fp > 0 else 0
    pp_r = pp_tp / total_gt if total_gt > 0 else 0
    pp_f1 = 2*pp_p*pp_r / (pp_p+pp_r) if pp_p+pp_r > 0 else 0
    all_ious = []
    for s in all_stats:
        all_ious.extend(s['pp_ious'])
    mean_iou = np.mean(all_ious) if all_ious else 0

    log.info(f"\nPost-processed (IoU>={cfg.eval_iou_thresh}):")
    log.info(f"  TP={pp_tp}, FP={pp_fp}, FN={pp_fn}")
    log.info(f"  Precision={pp_p:.4f}, Recall={pp_r:.4f}, F1={pp_f1:.4f}")
    log.info(f"  FP/volume={pp_fp/n_cases:.1f}")
    log.info(f"  Mean IoU (matched)={mean_iou:.4f}")

    # Post-processed evaluation (center)
    pp_c_tp = sum(s['pp_center_tp'] for s in all_stats)
    pp_c_fp = sum(s['pp_center_fp'] for s in all_stats)
    pp_c_fn = sum(s['pp_center_fn'] for s in all_stats)
    pp_c_p = pp_c_tp / (pp_c_tp + pp_c_fp) if pp_c_tp + pp_c_fp > 0 else 0
    pp_c_r = pp_c_tp / total_gt if total_gt > 0 else 0
    pp_c_f1 = 2*pp_c_p*pp_c_r / (pp_c_p+pp_c_r) if pp_c_p+pp_c_r > 0 else 0

    log.info(f"\nPost-processed (center dist<{cfg.eval_center_dist}):")
    log.info(f"  TP={pp_c_tp}, FP={pp_c_fp}, FN={pp_c_fn}")
    log.info(f"  Precision={pp_c_p:.4f}, Recall={pp_c_r:.4f}, F1={pp_c_f1:.4f}")

    # FP reduction ratio
    if raw_fp > 0:
        fp_reduction = (raw_fp - pp_fp) / raw_fp * 100
        log.info(f"\nFP reduction: {raw_fp} -> {pp_fp} ({fp_reduction:.1f}% removed)")

    # Save summary
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'config': {k: v for k, v in cfg.__dict__.items()
                   if not k.startswith('_')},
        'raw': {'tp': raw_tp, 'fp': raw_fp, 'fn': raw_fn,
                'precision': raw_p, 'recall': raw_r, 'f1': raw_f1},
        'postprocessed_iou': {'tp': pp_tp, 'fp': pp_fp, 'fn': pp_fn,
                              'precision': pp_p, 'recall': pp_r, 'f1': pp_f1,
                              'mean_iou': mean_iou},
        'postprocessed_center': {'tp': pp_c_tp, 'fp': pp_c_fp, 'fn': pp_c_fn,
                                 'precision': pp_c_p, 'recall': pp_c_r,
                                 'f1': pp_c_f1},
        'box_counts': {'raw': total_raw, 'after_score': total_after_score,
                       'after_size': total_after_size, 'clusters': total_clusters,
                       'final': total_final, 'gt': total_gt},
        'per_case': all_stats,
    }

    with open(out_dir / 'summary.json', 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(summary, f, indent=2, default=convert)

    with open(out_dir / 'summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    log.info(f"\nResults saved to {out_dir}")
    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process nnDetection predictions for FP reduction")

    # Paths
    parser.add_argument('--pred_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)

    # Pipeline parameters
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--min_support', type=int, default=1)
    parser.add_argument('--merge_dist', type=float, default=50.0)
    parser.add_argument('--merge_iou', type=float, default=0.01)
    parser.add_argument('--nms_iou', type=float, default=0.3)

    # Visualization
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis_dir', type=str, default=None)

    args = parser.parse_args()

    cfg = PostProcessConfig()

    # Override from CLI
    if args.pred_dir: cfg.pred_dir = args.pred_dir
    if args.data_dir: cfg.data_dir = args.data_dir
    if args.out_dir: cfg.out_dir = args.out_dir
    cfg.score_thresh = args.score_thresh
    cfg.min_support = args.min_support
    cfg.merge_center_dist = args.merge_dist
    cfg.merge_iou_thresh = args.merge_iou
    cfg.nms_iou_thresh = args.nms_iou
    cfg.visualize = args.visualize
    if args.vis_dir: cfg.vis_dir = args.vis_dir

    run_pipeline(cfg)


if __name__ == '__main__':
    main()

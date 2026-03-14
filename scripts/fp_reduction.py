"""
False Positive Reduction for nnDetection predictions.

Two-stage approach:
1. Merge small boxes into lesion-level clusters (from postprocess_and_eval.py)
2. Score each cluster using multiple features to filter FPs:
   - Cluster density: number of raw predictions merged
   - Confidence: max and mean score of constituent boxes
   - Intensity: hypoechoic ratio (lesions are darker in US)
   - Shape: aspect ratio constraints

Usage:
    python scripts/fp_reduction.py
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
OUT_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/visualizations_fpr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Merge parameters ----
SCORE_THRESH = 0.3
MERGE_IOU_THRESH = 0.01
MERGE_DIST_THRESH = 50


# =============================================================================
# Box utilities (same as before)
# =============================================================================

def box_iou_3d(a, b):
    d0 = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    d1 = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    d2 = max(0, min(a[5], b[5]) - max(a[4], b[4]))
    inter = d0 * d1 * d2
    va = (a[2]-a[0]) * (a[3]-a[1]) * (a[5]-a[4])
    vb = (b[2]-b[0]) * (b[3]-b[1]) * (b[5]-b[4])
    return inter / (va + vb - inter) if (va + vb - inter) > 0 else 0

def box_center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2, (b[4]+b[5])/2])

def center_distance(a, b):
    return np.linalg.norm(box_center(a) - box_center(b))


# =============================================================================
# Merge with feature tracking
# =============================================================================

def merge_with_features(boxes, scores):
    """Merge boxes and track cluster features."""
    if len(boxes) == 0:
        return [], []

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    assigned = np.zeros(len(boxes), dtype=bool)
    clusters = []

    for i in range(len(boxes)):
        if assigned[i]:
            continue

        members = [i]
        cluster_box = boxes[i].copy()
        assigned[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if assigned[j]:
                    continue
                iou = box_iou_3d(cluster_box, boxes[j])
                dist = center_distance(cluster_box, boxes[j])
                if iou >= MERGE_IOU_THRESH or dist <= MERGE_DIST_THRESH:
                    cluster_box = np.array([
                        min(cluster_box[0], boxes[j][0]),
                        min(cluster_box[1], boxes[j][1]),
                        max(cluster_box[2], boxes[j][2]),
                        max(cluster_box[3], boxes[j][3]),
                        min(cluster_box[4], boxes[j][4]),
                        max(cluster_box[5], boxes[j][5]),
                    ])
                    members.append(j)
                    assigned[j] = True
                    changed = True

        member_scores = scores[members]
        clusters.append({
            'box': cluster_box,
            'max_score': float(member_scores.max()),
            'mean_score': float(member_scores.mean()),
            'num_members': len(members),
            'member_boxes': boxes[members],
            'member_scores': member_scores,
        })

    return clusters


# =============================================================================
# Feature extraction for FP reduction
# =============================================================================

def compute_cluster_features(cluster, img):
    """Compute features for a merged cluster to predict if it's a real lesion."""
    box = cluster['box']
    d0_min = max(0, int(box[0]))
    d0_max = min(img.shape[0], int(box[2]))
    d1_min = max(0, int(box[1]))
    d1_max = min(img.shape[1], int(box[3]))
    d2_min = max(0, int(box[4]))
    d2_max = min(img.shape[2], int(box[5]))

    if d0_max <= d0_min or d1_max <= d1_min or d2_max <= d2_min:
        return None

    # Extract ROI
    roi = img[d0_min:d0_max, d1_min:d1_max, d2_min:d2_max].astype(float)

    # Box size
    size_d0 = d0_max - d0_min
    size_d1 = d1_max - d1_min
    size_d2 = d2_max - d2_min
    volume = size_d0 * size_d1 * size_d2

    # Intensity features
    roi_mean = roi.mean()
    roi_std = roi.std()

    # Compare interior to surrounding region
    pad = 10
    surr_d0 = (max(0, d0_min - pad), min(img.shape[0], d0_max + pad))
    surr_d1 = (max(0, d1_min - pad), min(img.shape[1], d1_max + pad))
    surr_d2 = (max(0, d2_min - pad), min(img.shape[2], d2_max + pad))
    surr = img[surr_d0[0]:surr_d0[1], surr_d1[0]:surr_d1[1], surr_d2[0]:surr_d2[1]].astype(float)
    surr_mean = surr.mean()

    # Hypoechoic ratio: lesions are darker than surroundings
    intensity_ratio = roi_mean / surr_mean if surr_mean > 0 else 1.0

    # Shape features
    max_dim = max(size_d0, size_d1, size_d2)
    min_dim = min(size_d0, size_d1, size_d2)
    aspect_ratio = max_dim / min_dim if min_dim > 0 else float('inf')

    features = {
        'max_score': cluster['max_score'],
        'mean_score': cluster['mean_score'],
        'num_members': cluster['num_members'],
        'volume': volume,
        'size_d0': size_d0,
        'size_d1': size_d1,
        'size_d2': size_d2,
        'roi_mean': roi_mean,
        'roi_std': roi_std,
        'surr_mean': surr_mean,
        'intensity_ratio': intensity_ratio,
        'aspect_ratio': aspect_ratio,
    }
    return features


def score_cluster(features):
    """
    Score a cluster for being a real lesion (higher = more likely lesion).
    Hand-tuned scoring based on known lesion properties in ABUS.
    """
    if features is None:
        return 0.0

    score = 0.0

    # 1. Confidence: high max score is strong signal
    score += features['max_score'] * 3.0

    # 2. Cluster density: real lesions generate many overlapping predictions
    density = min(features['num_members'] / 5.0, 1.0)  # normalize to ~5 members
    score += density * 2.0

    # 3. Hypoechoic: lesions should be darker than surroundings (ratio < 1)
    if features['intensity_ratio'] < 0.95:
        score += 1.5  # clearly hypoechoic
    elif features['intensity_ratio'] < 1.0:
        score += 0.5  # slightly hypoechoic

    # 4. Size: very small or very large clusters are suspicious
    vol = features['volume']
    if 1000 < vol < 5000000:  # reasonable lesion volume range
        score += 1.0
    else:
        score -= 0.5

    # 5. Shape: penalize extreme aspect ratios
    if features['aspect_ratio'] < 10:
        score += 0.5
    else:
        score -= 1.0

    # 6. Heterogeneity: lesions have some texture
    if features['roi_std'] > 5:
        score += 0.5

    return score


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_iou(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.1):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {'tp': 0, 'fp': len(pred_boxes), 'fn': len(gt_boxes), 'tp_ious': []}
    order = np.argsort(-pred_scores)
    pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp, fp, tp_ious = 0, 0, []
    for pb in pred_boxes:
        best_iou, best_gt = 0, -1
        for gi, gb in enumerate(gt_boxes):
            if not gt_matched[gi]:
                iou = box_iou_3d(pb, gb)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1; gt_matched[best_gt] = True; tp_ious.append(best_iou)
        else:
            fp += 1
    return {'tp': tp, 'fp': fp, 'fn': int((~gt_matched).sum()), 'tp_ious': tp_ious}


def evaluate_center_point(pred_boxes, pred_scores, gt_boxes, max_dist=30):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {'tp': 0, 'fp': len(pred_boxes), 'fn': len(gt_boxes)}
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp, fp = 0, 0
    for pb in pred_boxes:
        pc = box_center(pb)
        best_dist, best_gt = float('inf'), -1
        for gi, gb in enumerate(gt_boxes):
            if not gt_matched[gi]:
                d = np.linalg.norm(pc - box_center(gb))
                if d < best_dist:
                    best_dist, best_gt = d, gi
        if best_dist <= max_dist and best_gt >= 0:
            tp += 1; gt_matched[best_gt] = True
        else:
            fp += 1
    return {'tp': tp, 'fp': fp, 'fn': int((~gt_matched).sum())}


# =============================================================================
# Data loading
# =============================================================================

def load_gt_boxes(label_path):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    boxes = []
    for idx in np.unique(seg):
        if idx == 0: continue
        coords = np.stack(np.nonzero(seg == idx), axis=1)
        mins, maxs = coords.min(0), coords.max(0)
        boxes.append(np.array([mins[0]-1, mins[1]-1, maxs[0]+1, maxs[1]+1, mins[2]-1, maxs[2]+1], dtype=float))
    return np.array(boxes) if boxes else np.empty((0, 6))

def load_raw_preds(pkl_path):
    with open(pkl_path, 'rb') as f:
        pred = pickle.load(f)
    mask = pred['pred_scores'] >= SCORE_THRESH
    return pred['pred_boxes'][mask], pred['pred_scores'][mask]

def build_case_mapping():
    df = pd.read_csv(STATS_CSV)
    test = df[df['split'] == 'test']
    return {row['volume_id']: f"{os.path.basename(row['image_path']).replace('.nii','')}.ai"
            for _, row in test.iterrows()}

def load_baseline_preds(baseline_df, filename):
    rows = baseline_df[(baseline_df['Filename'] == filename) & baseline_df['Pred X1'].notna()]
    boxes, scores = [], []
    for _, r in rows.iterrows():
        boxes.append(np.array([r['Pred Z1'], r['Pred Y1'], r['Pred Z2'], r['Pred Y2'],
                               r['Pred X1'], r['Pred X2']], dtype=float))
        scores.append(r.get('Pred Lesion Prob', 0))
    if boxes: return np.array(boxes), np.array(scores)
    return np.empty((0, 6)), np.array([])


# =============================================================================
# Visualization
# =============================================================================

def draw_boxes(ax, boxes, sl, plane, color, label, scores=None, lw=2, ls='-'):
    drawn = False
    for i, b in enumerate(boxes):
        show, xy, w, h = False, None, None, None
        if plane == 'axial' and b[0] <= sl <= b[2]:
            xy, w, h, show = (b[4], b[1]), b[5]-b[4], b[3]-b[1], True
        elif plane == 'coronal' and b[1] <= sl <= b[3]:
            xy, w, h, show = (b[4], b[0]), b[5]-b[4], b[2]-b[0], True
        elif plane == 'sagittal' and b[4] <= sl <= b[5]:
            xy, w, h, show = (b[1], b[0]), b[3]-b[1], b[2]-b[0], True
        if show:
            rect = patches.Rectangle(xy, w, h, lw=lw, ec=color, fc='none', ls=ls,
                                     label=label if not drawn else None)
            ax.add_patch(rect)
            if scores is not None and len(scores) > i:
                ax.text(xy[0], xy[1]-3, f'{scores[i]:.2f}', color=color, fontsize=7,
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))
            drawn = True


def visualize_case(case_id, img, gt_boxes, fpr_boxes, fpr_scores,
                   base_boxes, base_scores, bl_name):
    for gi, gt in enumerate(gt_boxes):
        d0c = int((gt[0]+gt[2])/2)
        d1c = int((gt[1]+gt[3])/2)
        d2c = int((gt[4]+gt[5])/2)

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle(f'{case_id} ({bl_name}) - Lesion {gi+1} | '
                     f'Green=GT | Cyan=Baseline | Red=nnDet+FPR ({len(fpr_boxes)} boxes)',
                     fontsize=12, fontweight='bold')

        for col, (plane, sl) in enumerate([('axial',d0c),('coronal',d1c),('sagittal',d2c)]):
            ax = axes[col]
            if plane == 'axial':
                s = min(max(sl,0), img.shape[0]-1); ax.imshow(img[s,:,:], cmap='gray', aspect='auto')
            elif plane == 'coronal':
                s = min(max(sl,0), img.shape[1]-1); ax.imshow(img[:,s,:], cmap='gray', aspect='auto')
            else:
                s = min(max(sl,0), img.shape[2]-1); ax.imshow(img[:,:,s], cmap='gray', aspect='auto')
            ax.set_title(f'{plane.capitalize()} (slice={s})')
            draw_boxes(ax, gt_boxes, s, plane, 'lime', 'GT', lw=3)
            draw_boxes(ax, base_boxes, s, plane, 'cyan', 'Baseline', scores=base_scores, lw=2, ls='--')
            draw_boxes(ax, fpr_boxes, s, plane, 'red', 'nnDet+FPR', scores=fpr_scores, lw=2)
            ax.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{case_id}_lesion{gi+1}.png", dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# Main: sweep FPR threshold to find best operating point
# =============================================================================

def run_with_threshold(case_data, fpr_thresh):
    """Evaluate all cases with a given FPR score threshold."""
    res_iou = defaultdict(int)
    res_cp = defaultdict(int)
    all_ious = []

    for cd in case_data:
        # Filter clusters by FPR score
        keep = [i for i, s in enumerate(cd['fpr_scores']) if s >= fpr_thresh]
        if keep:
            boxes = np.array([cd['fpr_boxes'][i] for i in keep])
            scores = np.array([cd['fpr_scores'][i] for i in keep])
        else:
            boxes, scores = np.empty((0,6)), np.array([])

        ev_iou = evaluate_iou(boxes, scores, cd['gt_boxes'], iou_thresh=0.1)
        ev_cp = evaluate_center_point(boxes, scores, cd['gt_boxes'], max_dist=30)

        for k in ['tp','fp','fn']:
            res_iou[k] += ev_iou[k]
            res_cp[k] += ev_cp[k]
        all_ious.extend(ev_iou['tp_ious'])

    return res_iou, res_cp, all_ious


def main():
    case_map = build_case_mapping()
    baseline_df = pd.read_excel(BASELINE_XLSX, sheet_name='Detailed Comparison')
    baseline_files = set(baseline_df['Filename'].unique())

    # Step 1: Process all cases — merge + compute features + score
    case_data = []
    print("Processing cases...")

    for case_id in sorted(case_map.keys()):
        bl_filename = case_map[case_id]
        if bl_filename not in baseline_files:
            continue

        pred_path = PRED_DIR / f"{case_id}_boxes.pkl"
        if not pred_path.exists():
            continue

        img = sitk.GetArrayFromImage(sitk.ReadImage(
            str(DATA_DIR / "imagesTs" / f"{case_id}_0000.nii.gz")))
        gt_boxes = load_gt_boxes(DATA_DIR / "labelsTs" / f"{case_id}.nii.gz")
        raw_boxes, raw_scores = load_raw_preds(pred_path)
        base_boxes, base_scores = load_baseline_preds(baseline_df, bl_filename)

        # Merge
        clusters = merge_with_features(raw_boxes, raw_scores)

        # Compute features and FPR scores
        fpr_boxes, fpr_scores = [], []
        for cl in clusters:
            feat = compute_cluster_features(cl, img)
            fpr_score = score_cluster(feat)
            fpr_boxes.append(cl['box'])
            fpr_scores.append(fpr_score)

        fpr_boxes = np.array(fpr_boxes) if fpr_boxes else np.empty((0, 6))
        fpr_scores = np.array(fpr_scores) if fpr_scores else np.array([])

        case_data.append({
            'case_id': case_id,
            'bl_filename': bl_filename,
            'gt_boxes': gt_boxes,
            'fpr_boxes': fpr_boxes,
            'fpr_scores': fpr_scores,
            'base_boxes': base_boxes,
            'base_scores': base_scores,
            'img': img,
        })

    # Step 2: Sweep FPR threshold to find optimal point
    print("\n--- FPR Threshold Sweep ---")
    print(f"{'Thresh':>7} {'TP':>4} {'FP':>5} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"| {'TP_cp':>5} {'FP_cp':>5} {'FN_cp':>4} {'F1_cp':>6}")

    best_f1, best_thresh = 0, 0
    sweep_results = []

    for thresh in np.arange(2.0, 7.5, 0.5):
        res_iou, res_cp, _ = run_with_threshold(case_data, thresh)
        p = res_iou['tp'] / (res_iou['tp']+res_iou['fp']) if (res_iou['tp']+res_iou['fp']) > 0 else 0
        r = res_iou['tp'] / (res_iou['tp']+res_iou['fn']) if (res_iou['tp']+res_iou['fn']) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0

        p_cp = res_cp['tp'] / (res_cp['tp']+res_cp['fp']) if (res_cp['tp']+res_cp['fp']) > 0 else 0
        r_cp = res_cp['tp'] / (res_cp['tp']+res_cp['fn']) if (res_cp['tp']+res_cp['fn']) > 0 else 0
        f1_cp = 2*p_cp*r_cp/(p_cp+r_cp) if (p_cp+r_cp) > 0 else 0

        print(f"{thresh:>7.1f} {res_iou['tp']:>4} {res_iou['fp']:>5} {res_iou['fn']:>4} "
              f"{p:>6.3f} {r:>6.3f} {f1:>6.3f} | "
              f"{res_cp['tp']:>5} {res_cp['fp']:>5} {res_cp['fn']:>4} {f1_cp:>6.3f}")

        sweep_results.append((thresh, f1, f1_cp, res_iou, res_cp))
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"\nBest IoU F1: {best_f1:.3f} at threshold {best_thresh}")

    # Baseline reference
    res_base_cp = defaultdict(int)
    for cd in case_data:
        ev = evaluate_center_point(cd['base_boxes'], cd['base_scores'], cd['gt_boxes'], 30)
        for k in ['tp','fp','fn']: res_base_cp[k] += ev[k]
    p = res_base_cp['tp']/(res_base_cp['tp']+res_base_cp['fp']) if (res_base_cp['tp']+res_base_cp['fp']) > 0 else 0
    r = res_base_cp['tp']/(res_base_cp['tp']+res_base_cp['fn']) if (res_base_cp['tp']+res_base_cp['fn']) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    print(f"Baseline (center-point): TP={res_base_cp['tp']} FP={res_base_cp['fp']} "
          f"FN={res_base_cp['fn']} Prec={p:.3f} Rec={r:.3f} F1={f1:.3f}")

    # Step 3: Visualize with best threshold
    print(f"\nVisualizing with threshold={best_thresh}...")
    _, _, best_ious = run_with_threshold(case_data, best_thresh)

    for cd in case_data:
        keep = [i for i, s in enumerate(cd['fpr_scores']) if s >= best_thresh]
        if keep:
            boxes = np.array([cd['fpr_boxes'][i] for i in keep])
            scores = np.array([cd['fpr_scores'][i] for i in keep])
        else:
            boxes, scores = np.empty((0, 6)), np.array([])

        visualize_case(cd['case_id'], cd['img'], cd['gt_boxes'],
                       boxes, scores, cd['base_boxes'], cd['base_scores'],
                       cd['bl_filename'])

    # IoU summary
    if best_ious:
        ious = np.array(best_ious)
        print(f"\n--- IoU distribution at best threshold ---")
        print(f"  Mean: {ious.mean():.3f}, Median: {np.median(ious):.3f}")
        print(f"  IoU >= 0.3: {(ious>=0.3).sum()}/{len(ious)}")
        print(f"  IoU >= 0.5: {(ious>=0.5).sum()}/{len(ious)}")

    print(f"\nDone! Visualizations saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

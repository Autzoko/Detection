"""
Three-way comparison: GT vs Baseline vs nnDetection predictions.

Visualizes bounding boxes from all three sources on the same image slices.
- Green = GT (from instance segmentation masks)
- Blue = Baseline predictions (from Excel)
- Red = nnDetection predictions

Usage:
    python scripts/visualize_three_way.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


DATA_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted")
PRED_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/test_predictions")
STATS_CSV = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/dataset_statistics.csv")
BASELINE_XLSX = Path("/Users/langtian/Desktop/Prediction_vs_GT_Comparison.xlsx")
OUT_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/visualizations_3way")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NNDET_SCORE_THRESH = 0.3


def build_case_mapping():
    """Build mapping from case_id to baseline filename."""
    df = pd.read_csv(STATS_CSV)
    test = df[df['split'] == 'test']
    mapping = {}
    for _, row in test.iterrows():
        orig = os.path.basename(row['image_path']).replace('.nii', '')
        mapping[row['volume_id']] = f"{orig}.ai"
    return mapping


def load_baseline_preds(baseline_df, filename):
    """
    Load baseline predictions for a given filename.
    Returns boxes in nnDet format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
    where d0=z(500), d1=y(350), d2=x(1017).

    Baseline format: [X1, Y1, Z1, X2, Y2, Z2] where X=x, Y=y, Z=z.
    Mapping: nnDet = [Z1, Y1, Z2, Y2, X1, X2]
    """
    file_rows = baseline_df[baseline_df['Filename'] == filename]
    preds = file_rows[file_rows['Pred X1'].notna()]

    boxes = []
    match_types = []
    scores = []
    for _, row in preds.iterrows():
        x1, y1, z1 = row['Pred X1'], row['Pred Y1'], row['Pred Z1']
        x2, y2, z2 = row['Pred X2'], row['Pred Y2'], row['Pred Z2']
        # Convert to nnDet format [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
        box = [z1, y1, z2, y2, x1, x2]
        boxes.append(np.array(box, dtype=float))
        match_types.append(row['Match Type'])
        scores.append(row.get('Pred Lesion Prob', 0))

    if boxes:
        return np.array(boxes), match_types, np.array(scores)
    return np.empty((0, 6)), [], np.array([])


def load_gt_boxes(label_path):
    """Extract GT boxes from instance segmentation mask."""
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    boxes = []
    instances = np.unique(seg)
    instances = instances[instances > 0]
    for idx in instances:
        coords = np.stack(np.nonzero(seg == idx), axis=1)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        box = [mins[0] - 1, mins[1] - 1, maxs[0] + 1, maxs[1] + 1,
               mins[2] - 1, maxs[2] + 1]
        boxes.append(np.array(box, dtype=float))
    return np.array(boxes) if boxes else np.empty((0, 6))


def load_nndet_preds(pkl_path, score_thresh=0.3):
    """Load nnDetection predicted boxes and filter by score."""
    with open(pkl_path, 'rb') as f:
        pred = pickle.load(f)
    mask = pred['pred_scores'] >= score_thresh
    return pred['pred_boxes'][mask], pred['pred_scores'][mask]


def draw_boxes(ax, boxes, slice_idx, plane, color, label, scores=None,
               match_types=None, linewidth=2, linestyle='-'):
    """Draw bounding boxes on a 2D slice."""
    drawn = False
    for i, b in enumerate(boxes):
        d0_min, d1_min, d0_max, d1_max, d2_min, d2_max = b

        show = False
        if plane == 'axial' and d0_min <= slice_idx <= d0_max:
            xy = (d2_min, d1_min)
            w, h = d2_max - d2_min, d1_max - d1_min
            show = True
        elif plane == 'coronal' and d1_min <= slice_idx <= d1_max:
            xy = (d2_min, d0_min)
            w, h = d2_max - d2_min, d0_max - d0_min
            show = True
        elif plane == 'sagittal' and d2_min <= slice_idx <= d2_max:
            xy = (d1_min, d0_min)
            w, h = d1_max - d1_min, d0_max - d0_min
            show = True

        if show:
            rect = patches.Rectangle(
                xy, w, h, linewidth=linewidth, edgecolor=color,
                facecolor='none', linestyle=linestyle,
                label=label if not drawn else None)
            ax.add_patch(rect)
            # Score/match label
            txt = ''
            if scores is not None and len(scores) > i:
                txt = f'{scores[i]:.2f}'
            if match_types is not None and len(match_types) > i:
                mt = match_types[i]
                if 'TP' in mt:
                    txt += ' TP'
                elif 'FP' in mt:
                    txt += ' FP'
            if txt:
                ax.text(xy[0], xy[1] - 3, txt, color=color, fontsize=6,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))
            drawn = True


def visualize_case(case_id, baseline_filename, baseline_df):
    img_path = DATA_DIR / "imagesTs" / f"{case_id}_0000.nii.gz"
    label_path = DATA_DIR / "labelsTs" / f"{case_id}.nii.gz"
    pred_path = PRED_DIR / f"{case_id}_boxes.pkl"

    if not pred_path.exists() or not img_path.exists():
        print(f"  Skipping {case_id}: missing files")
        return

    # Load everything
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
    gt_boxes = load_gt_boxes(label_path)
    nndet_boxes, nndet_scores = load_nndet_preds(pred_path, NNDET_SCORE_THRESH)
    base_boxes, base_match_types, base_scores = load_baseline_preds(
        baseline_df, baseline_filename)

    print(f"  {case_id} ({baseline_filename}): img={img.shape}, "
          f"GT={len(gt_boxes)}, baseline={len(base_boxes)}, "
          f"nnDet(>={NNDET_SCORE_THRESH})={len(nndet_boxes)}")

    if len(gt_boxes) == 0:
        return

    # For each GT lesion, create a 3-view comparison
    for gi, gt_box in enumerate(gt_boxes):
        d0_c = int((gt_box[0] + gt_box[2]) / 2)
        d1_c = int((gt_box[1] + gt_box[3]) / 2)
        d2_c = int((gt_box[4] + gt_box[5]) / 2)

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle(
            f'{case_id} ({baseline_filename}) - Lesion {gi+1}\n'
            f'Green=GT  |  Blue=Baseline  |  Red=nnDetection',
            fontsize=13, fontweight='bold')

        views = [
            ('axial', d0_c, f'Axial (z={d0_c})', 'd2 (x)', 'd1 (y)'),
            ('coronal', d1_c, f'Coronal (y={d1_c})', 'd2 (x)', 'd0 (z)'),
            ('sagittal', d2_c, f'Sagittal (x={d2_c})', 'd1 (y)', 'd0 (z)'),
        ]

        for ax, (plane, sl, title, xlabel, ylabel) in zip(axes, views):
            if plane == 'axial':
                sl = min(max(sl, 0), img.shape[0] - 1)
                ax.imshow(img[sl, :, :], cmap='gray', aspect='auto')
            elif plane == 'coronal':
                sl = min(max(sl, 0), img.shape[1] - 1)
                ax.imshow(img[:, sl, :], cmap='gray', aspect='auto')
            else:
                sl = min(max(sl, 0), img.shape[2] - 1)
                ax.imshow(img[:, :, sl], cmap='gray', aspect='auto')

            ax.set_title(title, fontsize=11)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Draw GT (green, thick)
            draw_boxes(ax, gt_boxes, sl, plane, 'lime', 'GT', linewidth=3)
            # Draw baseline (blue, dashed)
            draw_boxes(ax, base_boxes, sl, plane, 'cyan', 'Baseline',
                       scores=base_scores, match_types=base_match_types,
                       linewidth=2, linestyle='--')
            # Draw nnDetection (red)
            draw_boxes(ax, nndet_boxes, sl, plane, 'red', 'nnDetection',
                       scores=nndet_scores, linewidth=2)

            ax.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        out_path = OUT_DIR / f"{case_id}_lesion{gi+1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # Build mapping
    case_map = build_case_mapping()
    print(f"Case mapping: {len(case_map)} test cases")

    # Load baseline data
    baseline_df = pd.read_excel(BASELINE_XLSX, sheet_name='Detailed Comparison')
    baseline_files = set(baseline_df['Filename'].unique())
    print(f"Baseline files: {len(baseline_files)}")

    # Process all cases
    for case_id in sorted(case_map.keys()):
        bl_filename = case_map[case_id]
        if bl_filename not in baseline_files:
            print(f"  {case_id}: baseline file {bl_filename} not found, skipping")
            continue
        visualize_case(case_id, bl_filename, baseline_df)

    print(f"\nDone! Visualizations saved to {OUT_DIR}")

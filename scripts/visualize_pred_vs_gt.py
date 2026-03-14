"""
Visualize nnDetection predicted bounding boxes vs GT bounding boxes.

nnDetection box format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
Array shape from SimpleITK: (d0, d1, d2) = (z, y, x) in physical terms.

Usage:
    python scripts/visualize_pred_vs_gt.py
"""

import pickle
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


DATA_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted")
PRED_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/test_predictions")
OUT_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_THRESH = 0.3  # only show predictions above this threshold
NUM_CASES = None    # None = all cases


def load_gt_boxes(label_path):
    """Extract GT boxes from instance segmentation mask."""
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    dim = 3
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


def load_pred_boxes(pkl_path, score_thresh=0.3):
    """Load predicted boxes and filter by score."""
    with open(pkl_path, 'rb') as f:
        pred = pickle.load(f)
    mask = pred['pred_scores'] >= score_thresh
    return pred['pred_boxes'][mask], pred['pred_scores'][mask]


def draw_boxes_on_slice(ax, boxes, slice_idx, plane, color, label, scores=None):
    """
    Draw bounding boxes on a 2D slice.

    plane: 'axial' (fix d0, show d2 vs d1), 'coronal' (fix d1, show d2 vs d0),
           'sagittal' (fix d2, show d1 vs d0)

    Box format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
    """
    drawn = False
    for i, b in enumerate(boxes):
        d0_min, d1_min, d0_max, d1_max, d2_min, d2_max = b

        if plane == 'axial':  # fix d0, axes: x=d2, y=d1
            if d0_min <= slice_idx <= d0_max:
                rect = patches.Rectangle(
                    (d2_min, d1_min), d2_max - d2_min, d1_max - d1_min,
                    linewidth=2, edgecolor=color, facecolor='none',
                    label=label if not drawn else None)
                ax.add_patch(rect)
                if scores is not None:
                    ax.text(d2_min, d1_min - 2, f'{scores[i]:.2f}',
                            color=color, fontsize=7, fontweight='bold')
                drawn = True

        elif plane == 'coronal':  # fix d1, axes: x=d2, y=d0
            if d1_min <= slice_idx <= d1_max:
                rect = patches.Rectangle(
                    (d2_min, d0_min), d2_max - d2_min, d0_max - d0_min,
                    linewidth=2, edgecolor=color, facecolor='none',
                    label=label if not drawn else None)
                ax.add_patch(rect)
                if scores is not None:
                    ax.text(d2_min, d0_min - 2, f'{scores[i]:.2f}',
                            color=color, fontsize=7, fontweight='bold')
                drawn = True

        elif plane == 'sagittal':  # fix d2, axes: x=d1, y=d0
            if d2_min <= slice_idx <= d2_max:
                rect = patches.Rectangle(
                    (d1_min, d0_min), d1_max - d1_min, d0_max - d0_min,
                    linewidth=2, edgecolor=color, facecolor='none',
                    label=label if not drawn else None)
                ax.add_patch(rect)
                if scores is not None:
                    ax.text(d1_min, d0_min - 2, f'{scores[i]:.2f}',
                            color=color, fontsize=7, fontweight='bold')
                drawn = True


def visualize_case(case_id):
    img_path = DATA_DIR / "imagesTs" / f"{case_id}_0000.nii.gz"
    label_path = DATA_DIR / "labelsTs" / f"{case_id}.nii.gz"
    pred_path = PRED_DIR / f"{case_id}_boxes.pkl"

    if not pred_path.exists():
        print(f"Skipping {case_id}: no predictions found")
        return

    # Load data
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))  # (d0, d1, d2)
    gt_boxes = load_gt_boxes(label_path)
    pred_boxes, pred_scores = load_pred_boxes(pred_path, SCORE_THRESH)

    print(f"\n{case_id}: image={img.shape}, GT boxes={len(gt_boxes)}, "
          f"pred boxes (score>={SCORE_THRESH})={len(pred_boxes)}")

    if len(gt_boxes) == 0:
        print(f"  No GT boxes, skipping")
        return

    # For each GT box, show 3 views centered on the GT box center
    for gi, gt_box in enumerate(gt_boxes):
        d0_min, d1_min, d0_max, d1_max, d2_min, d2_max = gt_box
        d0_c = int((d0_min + d0_max) / 2)
        d1_c = int((d1_min + d1_max) / 2)
        d2_c = int((d2_min + d2_max) / 2)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{case_id} - GT Lesion {gi+1} '
                     f'(center: d0={d0_c}, d1={d1_c}, d2={d2_c})',
                     fontsize=14, fontweight='bold')

        # Axial: fix d0 at center, show d2 (horizontal) vs d1 (vertical)
        ax = axes[0]
        ax.imshow(img[d0_c, :, :], cmap='gray', aspect='auto')
        ax.set_title(f'Axial (d0={d0_c})')
        ax.set_xlabel('d2 (x physical)')
        ax.set_ylabel('d1 (y physical)')
        draw_boxes_on_slice(ax, gt_boxes, d0_c, 'axial', 'lime', 'GT')
        draw_boxes_on_slice(ax, pred_boxes, d0_c, 'axial', 'red', 'Pred', pred_scores)
        ax.legend(loc='upper right', fontsize=8)

        # Coronal: fix d1 at center, show d2 (horizontal) vs d0 (vertical)
        ax = axes[1]
        ax.imshow(img[:, d1_c, :], cmap='gray', aspect='auto')
        ax.set_title(f'Coronal (d1={d1_c})')
        ax.set_xlabel('d2 (x physical)')
        ax.set_ylabel('d0 (z physical)')
        draw_boxes_on_slice(ax, gt_boxes, d1_c, 'coronal', 'lime', 'GT')
        draw_boxes_on_slice(ax, pred_boxes, d1_c, 'coronal', 'red', 'Pred', pred_scores)
        ax.legend(loc='upper right', fontsize=8)

        # Sagittal: fix d2 at center, show d1 (horizontal) vs d0 (vertical)
        ax = axes[2]
        ax.imshow(img[:, :, d2_c], cmap='gray', aspect='auto')
        ax.set_title(f'Sagittal (d2={d2_c})')
        ax.set_xlabel('d1 (y physical)')
        ax.set_ylabel('d0 (z physical)')
        draw_boxes_on_slice(ax, gt_boxes, d2_c, 'sagittal', 'lime', 'GT')
        draw_boxes_on_slice(ax, pred_boxes, d2_c, 'sagittal', 'red', 'Pred', pred_scores)
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        out_path = OUT_DIR / f"{case_id}_lesion{gi+1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    # Get case IDs from prediction dir
    case_ids = sorted([p.stem.replace('_boxes', '')
                       for p in PRED_DIR.glob('*_boxes.pkl')])
    print(f"Found {len(case_ids)} cases")

    for case_id in (case_ids[:NUM_CASES] if NUM_CASES else case_ids):
        visualize_case(case_id)

    print(f"\nDone! Visualizations saved to {OUT_DIR}")

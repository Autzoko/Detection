"""
Per-case visualization: GT vs baseline vs post-processed predictions.

For each case, generates a figure with slices through each GT lesion center,
showing GT boxes (green), baseline predictions (red), and post-processed
predictions (blue) overlaid on the volume.
"""

import json
import os
import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml


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
    return np.array([
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
        (box[4] + box[5]) / 2.0,
    ])


def iou_3d(box1, box2):
    z1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1]); x1 = max(box1[4], box2[4])
    z2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3]); x2 = min(box1[5], box2[5])
    inter = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    vol1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    vol2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.1):
    """Returns (tp_indices, fp_indices) into pred arrays."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_pred == 0:
        return [], []
    sorted_idx = np.argsort(-pred_scores)
    gt_matched = [False] * n_gt
    tp_idx, fp_idx = [], []
    for pi in sorted_idx:
        best_iou, best_gi = 0, -1
        for gi, gb in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            ov = iou_3d(pred_boxes[pi], gb)
            if ov > best_iou:
                best_iou = ov
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            tp_idx.append(pi)
            gt_matched[best_gi] = True
        else:
            fp_idx.append(pi)
    return tp_idx, fp_idx


def draw_box_on_slice(ax, box, z_slice, color, linestyle="-", linewidth=2, label=None):
    """Draw a box on an axial slice if the slice intersects the box.
    box: [z_min, y_min, z_max, y_max, x_min, x_max]
    """
    z_min, y_min, z_max, y_max, x_min, x_max = box[:6]
    if z_slice < z_min or z_slice >= z_max:
        return False
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                          linewidth=linewidth, edgecolor=color, facecolor="none",
                          linestyle=linestyle, label=label)
    ax.add_patch(rect)
    return True


def draw_box_on_coronal(ax, box, y_slice, color, linestyle="-", linewidth=2, label=None):
    """Draw a box on a coronal slice (y fixed). Axes: x horizontal, z vertical."""
    z_min, y_min, z_max, y_max, x_min, x_max = box[:6]
    if y_slice < y_min or y_slice >= y_max:
        return False
    rect = plt.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                          linewidth=linewidth, edgecolor=color, facecolor="none",
                          linestyle=linestyle, label=label)
    ax.add_patch(rect)
    return True


def draw_box_on_sagittal(ax, box, x_slice, color, linestyle="-", linewidth=2, label=None):
    """Draw a box on a sagittal slice (x fixed). Axes: y horizontal, z vertical."""
    z_min, y_min, z_max, y_max, x_min, x_max = box[:6]
    if x_slice < x_min or x_slice >= x_max:
        return False
    rect = plt.Rectangle((y_min, z_min), y_max - y_min, z_max - z_min,
                          linewidth=linewidth, edgecolor=color, facecolor="none",
                          linestyle=linestyle, label=label)
    ax.add_patch(rect)
    return True


def visualize_case(case_id, volume, gt_boxes, baseline_boxes, baseline_scores,
                   postproc_boxes, postproc_scores, save_dir, score_thresh=0.3):
    """Create a multi-panel figure for one case.

    For each GT lesion, show 3 views (axial, coronal, sagittal) through its center.
    Also show an overview panel with all FPs on a representative slice.
    """
    # Filter baseline by score threshold for fair comparison
    if len(baseline_scores) > 0:
        mask = baseline_scores >= score_thresh
        bl_boxes = baseline_boxes[mask]
        bl_scores = baseline_scores[mask]
    else:
        bl_boxes = baseline_boxes
        bl_scores = baseline_scores

    pp_boxes = postproc_boxes
    pp_scores = postproc_scores

    # Match baseline and postproc to GT
    gt_box_list = [g["box"] for g in gt_boxes]
    bl_tp, bl_fp = match_preds_to_gt(bl_boxes, bl_scores, gt_box_list)
    pp_tp, pp_fp = match_preds_to_gt(pp_boxes, pp_scores, gt_box_list)

    n_gt = len(gt_boxes)
    n_lesion_rows = max(n_gt, 1)

    # Normalize volume for display
    p1, p99 = np.percentile(volume[volume > 0], [1, 99]) if (volume > 0).any() else (0, 1)
    disp = np.clip((volume.astype(float) - p1) / (p99 - p1 + 1e-8), 0, 1)

    fig, axes = plt.subplots(n_lesion_rows, 3, figsize=(18, 5 * n_lesion_rows))
    if n_lesion_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{case_id}  |  GT: {n_gt}  |  "
        f"Baseline (t>={score_thresh}): {len(bl_boxes)} pred ({len(bl_tp)} TP, {len(bl_fp)} FP)  |  "
        f"Post-proc: {len(pp_boxes)} pred ({len(pp_tp)} TP, {len(pp_fp)} FP)",
        fontsize=13, fontweight="bold", y=1.02
    )

    view_funcs = [
        ("Axial (z)", lambda s: disp[s, :, :], draw_box_on_slice, 0),
        ("Coronal (y)", lambda s: disp[:, s, :], draw_box_on_coronal, 1),
        ("Sagittal (x)", lambda s: disp[:, :, s], draw_box_on_sagittal, 2),
    ]

    for row_i in range(n_lesion_rows):
        if row_i < n_gt:
            gt = gt_boxes[row_i]
            center = box_center(gt["box"])
            slice_indices = [int(center[0]), int(center[1]), int(center[2])]
        else:
            # No GT — show volume center
            slice_indices = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]

        for col_i, (view_name, get_slice, draw_func, center_idx) in enumerate(view_funcs):
            ax = axes[row_i, col_i]
            s = slice_indices[center_idx]
            img = get_slice(s)

            if col_i == 0:
                ax.imshow(img, cmap="gray", aspect="auto")
            else:
                ax.imshow(img, cmap="gray", aspect="auto")

            # Draw GT boxes
            if row_i < n_gt:
                draw_func(ax, gt["box"], s, color="lime", linewidth=2.5,
                          label="GT" if col_i == 0 else None)

            # Draw baseline predictions (red = FP, orange = TP)
            for pi in range(len(bl_boxes)):
                is_tp = pi in bl_tp
                color = "orange" if is_tp else "red"
                lw = 1.5 if is_tp else 1.0
                ls = "-" if is_tp else "--"
                lbl = None
                if col_i == 0 and pi == 0:
                    lbl = f"Baseline ({'TP' if is_tp else 'FP'})"
                draw_func(ax, bl_boxes[pi], s, color=color, linewidth=lw,
                          linestyle=ls, label=lbl)

            # Draw post-processed predictions (cyan = TP, blue = FP)
            for pi in range(len(pp_boxes)):
                is_tp = pi in pp_tp
                color = "cyan" if is_tp else "dodgerblue"
                lw = 2.0 if is_tp else 1.0
                ls = "-" if is_tp else ":"
                lbl = None
                if col_i == 0 and pi == 0:
                    lbl = f"PostProc ({'TP' if is_tp else 'FP'})"
                draw_func(ax, pp_boxes[pi], s, color=color, linewidth=lw,
                          linestyle=ls, label=lbl)

            title = f"{view_name} slice={s}"
            if row_i < n_gt:
                title = f"Lesion {row_i + 1} — {title}"
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    # Build legend
    legend_handles = [
        mpatches.Patch(edgecolor="lime", facecolor="none", linewidth=2.5, label="GT"),
        mpatches.Patch(edgecolor="orange", facecolor="none", linewidth=1.5, label="Baseline TP"),
        mpatches.Patch(edgecolor="red", facecolor="none", linewidth=1, linestyle="--", label="Baseline FP"),
        mpatches.Patch(edgecolor="cyan", facecolor="none", linewidth=2, label="PostProc TP"),
        mpatches.Patch(edgecolor="dodgerblue", facecolor="none", linewidth=1, linestyle=":", label="PostProc FP"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(save_dir, f"{case_id}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    config_path = Path(__file__).parent / "patch_classifier" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pred_dir = cfg["paths"]["test_predictions_dir"]
    images_dir = cfg["paths"]["test_images_dir"]
    labels_dir = cfg["paths"]["test_labels_dir"]
    pp_cfg = cfg["postprocessing"]

    # Find the post-processed predictions directory (most recent)
    pp_base = pp_cfg["output_dir"]
    pp_subdirs = [d for d in Path(pp_base).iterdir() if d.is_dir() and d.name.startswith("cluster")]
    if not pp_subdirs:
        print("No post-processed prediction directories found.")
        return
    pp_pred_dir = sorted(pp_subdirs)[-1]
    print(f"Using post-processed predictions from: {pp_pred_dir}")

    save_dir = os.path.join(pp_base, "case_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    score_thresh = 0.3  # baseline filter threshold

    pred_files = sorted(Path(pred_dir).glob("*_boxes.pkl"))
    case_ids = [f.stem.replace("_boxes", "") for f in pred_files]
    print(f"Visualizing {len(case_ids)} cases ...")

    for ci, case_id in enumerate(case_ids):
        # Load volume
        img_path = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
        img_sitk = sitk.ReadImage(str(img_path))
        volume = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)

        # Load GT
        gt_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        json_path = os.path.join(labels_dir, f"{case_id}.json")
        gt_list = extract_gt_boxes(gt_path, json_path)

        # Load baseline predictions
        with open(os.path.join(pred_dir, f"{case_id}_boxes.pkl"), "rb") as f:
            bl_pred = pickle.load(f)
        bl_boxes = bl_pred["pred_boxes"]
        bl_scores = bl_pred["pred_scores"]

        # Load post-processed predictions
        pp_pkl = os.path.join(pp_pred_dir, f"{case_id}_boxes.pkl")
        if os.path.exists(pp_pkl):
            with open(pp_pkl, "rb") as f:
                pp_pred = pickle.load(f)
            pp_boxes = pp_pred["pred_boxes"]
            pp_scores = pp_pred["pred_scores"]
        else:
            pp_boxes = np.zeros((0, 6))
            pp_scores = np.zeros(0)

        path = visualize_case(
            case_id, volume, gt_list, bl_boxes, bl_scores,
            pp_boxes, pp_scores, save_dir, score_thresh=score_thresh
        )
        print(f"  [{ci + 1}/{len(case_ids)}] {case_id} — {len(gt_list)} GT, "
              f"{(bl_scores >= score_thresh).sum()} baseline, {len(pp_boxes)} postproc → {path}")

    # Create a summary montage: cases sorted by FP reduction
    print(f"\nAll case visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()

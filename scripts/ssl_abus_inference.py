"""
SSL_ABUS inference on our ABUS data.

SSL_ABUS architecture:
  - DATTR2U-Net (backbone): 3D segmentation U-Net, input (B,1,H,W,D) -> (B,1,H,W,D)
  - Detection_model (head): per-slice detection, input (B,1,375,250,32) -> (B,32,5)
    where output[:,:,0]=probability, output[:,:,1:5]=[x,y,w,h]

NOTE: The pretrained weights from Google Drive are likely BACKBONE-ONLY
(self-supervised pretrained via inpainting/denoising on TDSC-ABUS).
The detection head was trained separately and may not be included.

This script supports two modes:
  1. "segmentation" — Use backbone only for segmentation-based detection
     (threshold segmentation output, find connected components -> 3D boxes)
  2. "detection" — Use full combined model (backbone + detection head)
     if detection weights are available

Our data: (500, 350, 1017) = (z, y, x), spacing [1.0, 3.0, 1.0]
TDSC-ABUS patches: (375, 250, 32), spacing ~(0.2, 0.2, 0.5)

Usage:
    python scripts/ssl_abus_inference.py --mode segmentation
    python scripts/ssl_abus_inference.py --mode detection
"""

import os
import sys
import argparse
import pickle
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import ndimage

# Add SSL_ABUS to path
SSL_ABUS_DIR = Path("/Users/langtian/Desktop/NYU/MS Thesis/SSL_ABUS")
sys.path.insert(0, str(SSL_ABUS_DIR))

from model.DATTR2_UNET import DoubleATTR2U_Net
from model.DETECTION import Detection_model

# === Configuration ===
DATA_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted")
WEIGHTS_DIR = SSL_ABUS_DIR / "weights"
OUT_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/ssl_abus_predictions")
VIS_DIR = Path("/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/visualizations_ssl_abus")

# SSL_ABUS expected patch size
PATCH_SIZE = (375, 250, 32)  # (H, W, D) as used by the model
# Backbone features
BACKBONE_FEATURES = [32, 64, 128, 256, 512]
DETECTION_FEATURES = [16, 32, 64]

# Segmentation mode thresholds
SEG_THRESHOLD = 0.5
MIN_VOLUME_VOXELS = 100  # minimum voxels for a detection


def load_backbone(weight_path, device):
    """Load the DATTR2U-Net backbone."""
    model = DoubleATTR2U_Net(in_ch=1, out_ch=1, features=BACKBONE_FEATURES, t=2)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded backbone from {weight_path}")
    return model


def load_combined_model(backbone_path, detection_path, device):
    """Load backbone + detection head."""
    backbone = DoubleATTR2U_Net(in_ch=1, out_ch=1, features=BACKBONE_FEATURES, t=2)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))

    det_head = Detection_model(in_ch=1, out_ch=1, features=DETECTION_FEATURES, threshold=0.5)
    if detection_path and Path(detection_path).exists():
        det_head.load_state_dict(torch.load(detection_path, map_location=device))
        print(f"Loaded detection head from {detection_path}")

    combined = nn.Sequential(backbone, det_head).to(device)
    combined.eval()
    print(f"Loaded combined model")
    return combined


def normalize_volume(vol):
    """Normalize volume to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 0:
        return (vol - vmin) / (vmax - vmin)
    return np.zeros_like(vol)


def extract_patches_sliding_window(volume, patch_size=PATCH_SIZE, stride_d=16):
    """
    Extract patches from volume using sliding window along the depth (last) axis.

    volume: (D0, D1, D2) = (z, y, x) from SimpleITK
    patch_size: (H, W, D) = (375, 250, 32) as expected by SSL_ABUS

    Strategy:
    - Resize (z, y) dimensions to (375, 250)
    - Slide along x dimension with window size 32

    Returns: list of (patch, x_start, x_end) tuples
    """
    z_dim, y_dim, x_dim = volume.shape

    # Resize z,y to match SSL_ABUS expected H,W
    # Use scipy zoom for 3D resize
    target_h, target_w, target_d = patch_size
    zoom_z = target_h / z_dim
    zoom_y = target_w / y_dim

    # Resize only z and y, keep x as-is for sliding window
    resized = ndimage.zoom(volume, (zoom_z, zoom_y, 1.0), order=1)
    print(f"  Resized: {volume.shape} -> {resized.shape} (zoom_z={zoom_z:.3f}, zoom_y={zoom_y:.3f})")

    patches = []
    x_total = resized.shape[2]

    # Sliding window along x (depth) dimension
    x_start = 0
    while x_start < x_total:
        x_end = min(x_start + target_d, x_total)

        if x_end - x_start < target_d:
            # Last patch: pad or start from end
            if x_total >= target_d:
                x_start = x_total - target_d
                x_end = x_total
            else:
                # Volume too small, pad with zeros
                patch = np.zeros((target_h, target_w, target_d), dtype=np.float32)
                patch[:, :, :x_total] = resized[:target_h, :target_w, :]
                patches.append((patch, 0, x_total))
                break

        patch = resized[:target_h, :target_w, x_start:x_end]

        # Ensure exact size (may need minor padding due to rounding)
        if patch.shape[0] < target_h or patch.shape[1] < target_w:
            padded = np.zeros((target_h, target_w, target_d), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded

        patches.append((patch, x_start, x_end))

        if x_end >= x_total:
            break
        x_start += stride_d

    return patches, (zoom_z, zoom_y)


def segmentation_inference(model, volume, device, threshold=SEG_THRESHOLD):
    """
    Use backbone segmentation for detection.
    Run sliding window, threshold output, find connected components.

    Returns boxes in nnDet format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
    and scores.
    """
    volume_norm = normalize_volume(volume.astype(np.float32))
    patches, (zoom_z, zoom_y) = extract_patches_sliding_window(volume_norm)
    print(f"  Extracted {len(patches)} patches")

    # Reconstruct full segmentation map in resized space
    resized_shape = (PATCH_SIZE[0], PATCH_SIZE[1], volume.shape[2])
    seg_map = np.zeros(resized_shape, dtype=np.float32)
    count_map = np.zeros(resized_shape, dtype=np.float32)

    with torch.no_grad():
        for patch, x_start, x_end in patches:
            # (H, W, D) -> (1, 1, H, W, D) for model input
            tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)
            output = torch.sigmoid(output)  # ensure [0,1]
            pred = output.squeeze().cpu().numpy()  # (H, W, D)

            seg_map[:, :, x_start:x_end] += pred[:, :, :x_end - x_start]
            count_map[:, :, x_start:x_end] += 1.0

    # Average overlapping predictions
    count_map[count_map == 0] = 1
    seg_map /= count_map

    # Threshold
    binary = (seg_map > threshold).astype(np.uint8)
    print(f"  Segmentation: {binary.sum()} positive voxels out of {binary.size}")

    # Connected components
    labeled, num_features = ndimage.label(binary)
    print(f"  Found {num_features} connected components")

    boxes = []
    scores = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) < MIN_VOLUME_VOXELS:
            continue

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # Average segmentation confidence in this region
        region_mask = labeled == i
        score = float(seg_map[region_mask].mean())

        # Convert from resized coordinates back to original
        # resized: (375, 250, x_original) -> original: (500, 350, 1017)
        d0_min = mins[0] / zoom_z
        d0_max = maxs[0] / zoom_z
        d1_min = mins[1] / zoom_y
        d1_max = maxs[1] / zoom_y
        d2_min = float(mins[2])  # x dimension was not resized
        d2_max = float(maxs[2])

        box = [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
        boxes.append(np.array(box, dtype=float))
        scores.append(score)

    if boxes:
        return np.array(boxes), np.array(scores)
    return np.empty((0, 6)), np.array([])


def detection_inference(model, volume, device):
    """
    Use full combined model (backbone + detection head).
    The detection head outputs per-slice [prob, x, y, w, h].

    Returns boxes in nnDet format and scores.
    """
    volume_norm = normalize_volume(volume.astype(np.float32))
    patches, (zoom_z, zoom_y) = extract_patches_sliding_window(volume_norm)
    print(f"  Extracted {len(patches)} patches")

    all_boxes = []
    all_scores = []

    with torch.no_grad():
        for patch, x_start, x_end in patches:
            tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)  # (1, 32, 5)
            output = output.squeeze(0).cpu().numpy()  # (32, 5)

            for s in range(output.shape[0]):
                prob = output[s, 0]
                if prob < 0.5:
                    continue

                x, y, w, h = output[s, 1:5]

                # The detection head outputs 2D bbox in resized patch space
                # Convert to original volume coordinates
                # Patch space: (375 = z_resized, 250 = y_resized, slice_in_x)
                # x,y,w,h are in the 2D plane of the resized patch

                # Map back: bbox center/size in resized -> original
                d1_min = y / zoom_y  # y -> d1
                d1_max = (y + h) / zoom_y
                d0_min = x / zoom_z  # x -> d0
                d0_max = (x + w) / zoom_z

                # Depth: slice index maps to x_start + s
                actual_x = x_start + s
                d2_min = float(actual_x - 1)
                d2_max = float(actual_x + 1)

                box = [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
                all_boxes.append(np.array(box, dtype=float))
                all_scores.append(float(prob))

    # Merge nearby per-slice boxes into 3D boxes
    if all_boxes:
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        boxes, scores = merge_slice_detections(boxes, scores)
        return boxes, scores
    return np.empty((0, 6)), np.array([])


def merge_slice_detections(boxes, scores, iou_thresh=0.1, dist_thresh=30):
    """Merge per-slice 2D detections into 3D bounding boxes."""
    if len(boxes) == 0:
        return boxes, scores

    merged_boxes = []
    merged_scores = []
    used = np.zeros(len(boxes), dtype=bool)

    # Sort by score descending
    order = np.argsort(-scores)

    for idx in order:
        if used[idx]:
            continue

        used[idx] = True
        cluster = [idx]

        # Find all overlapping/nearby boxes
        for jdx in order:
            if used[jdx]:
                continue
            # Check spatial proximity
            bi, bj = boxes[idx], boxes[jdx]
            center_i = np.array([(bi[0]+bi[2])/2, (bi[1]+bi[3])/2, (bi[4]+bi[5])/2])
            center_j = np.array([(bj[0]+bj[2])/2, (bj[1]+bj[3])/2, (bj[4]+bj[5])/2])
            dist = np.linalg.norm(center_i - center_j)
            if dist < dist_thresh:
                cluster.append(jdx)
                used[jdx] = True

        # Merge cluster into single 3D box
        cluster_boxes = boxes[cluster]
        merged_box = [
            cluster_boxes[:, 0].min(),
            cluster_boxes[:, 1].min(),
            cluster_boxes[:, 2].max(),
            cluster_boxes[:, 3].max(),
            cluster_boxes[:, 4].min(),
            cluster_boxes[:, 5].max(),
        ]
        merged_score = scores[cluster].max()
        merged_boxes.append(np.array(merged_box, dtype=float))
        merged_scores.append(merged_score)

    return np.array(merged_boxes), np.array(merged_scores)


def compute_iou_3d(box1, box2):
    """Compute 3D IoU between two boxes in nnDet format."""
    d0_min = max(box1[0], box2[0])
    d1_min = max(box1[1], box2[1])
    d0_max = min(box1[2], box2[2])
    d1_max = min(box1[3], box2[3])
    d2_min = max(box1[4], box2[4])
    d2_max = min(box1[5], box2[5])

    inter = max(0, d0_max - d0_min) * max(0, d1_max - d1_min) * max(0, d2_max - d2_min)
    vol1 = (box1[2]-box1[0]) * (box1[3]-box1[1]) * (box1[5]-box1[4])
    vol2 = (box2[2]-box2[0]) * (box2[3]-box2[1]) * (box2[5]-box2[4])
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0


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
        box = [mins[0]-1, mins[1]-1, maxs[0]+1, maxs[1]+1, mins[2]-1, maxs[2]+1]
        boxes.append(np.array(box, dtype=float))
    return np.array(boxes) if boxes else np.empty((0, 6))


def evaluate_predictions(all_results, iou_thresh=0.1):
    """Evaluate all predictions against GT."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious = []

    for case_id, pred_boxes, pred_scores, gt_boxes in all_results:
        matched_gt = set()
        matched_pred = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Sort predictions by score
            order = np.argsort(-pred_scores)
            for pi in order:
                best_iou = 0
                best_gi = -1
                for gi in range(len(gt_boxes)):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou_3d(pred_boxes[pi], gt_boxes[gi])
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= iou_thresh and best_gi >= 0:
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)
                    ious.append(best_iou)

        tp = len(matched_pred)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"  {case_id}: pred={len(pred_boxes)}, GT={len(gt_boxes)}, "
              f"TP={tp}, FP={fp}, FN={fn}")

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    mean_iou = np.mean(ious) if ious else 0

    print(f"\n=== SSL_ABUS Evaluation (IoU>={iou_thresh}) ===")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f"Mean IoU (matched)={mean_iou:.4f}")
    print(f"FP per volume={total_fp / max(len(all_results), 1):.2f}")

    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_iou': mean_iou, 'ious': ious,
    }


def find_weights():
    """Find available weight files."""
    if not WEIGHTS_DIR.exists():
        return None, None

    pth_files = list(WEIGHTS_DIR.glob("*.pth"))
    pt_files = list(WEIGHTS_DIR.glob("*.pt"))
    all_weights = pth_files + pt_files

    if not all_weights:
        return None, None

    print(f"Found weight files: {[f.name for f in all_weights]}")

    # Try to identify backbone vs detection weights by name
    backbone_path = None
    detection_path = None

    for f in all_weights:
        name = f.stem.lower()
        if any(k in name for k in ['datt', 'backbone', 'unet', 'pretrain', 'multi']):
            backbone_path = f
        elif any(k in name for k in ['detect', 'combined', 'full']):
            detection_path = f

    # If only one file, assume it's the backbone
    if backbone_path is None and len(all_weights) == 1:
        backbone_path = all_weights[0]
    elif backbone_path is None and len(all_weights) > 0:
        # Pick the largest file as backbone (U-Net is bigger than detection head)
        all_weights.sort(key=lambda f: f.stat().st_size, reverse=True)
        backbone_path = all_weights[0]
        if len(all_weights) > 1:
            detection_path = all_weights[1]

    return backbone_path, detection_path


def main():
    parser = argparse.ArgumentParser(description="SSL_ABUS inference on ABUS data")
    parser.add_argument("--mode", choices=["segmentation", "detection"],
                        default="segmentation",
                        help="Inference mode: segmentation (backbone only) or detection (full)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Segmentation threshold (segmentation mode)")
    parser.add_argument("--num_cases", type=int, default=None,
                        help="Number of cases to process (None=all)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Find weights
    backbone_path, detection_path = find_weights()
    if backbone_path is None:
        print(f"\nERROR: No weight files found in {WEIGHTS_DIR}/")
        print(f"Please download weights from:")
        print(f"  https://drive.google.com/drive/folders/14XkuninPXx0IlDigjaMjILmb-pzdbxnb")
        print(f"And save .pth files to {WEIGHTS_DIR}/")
        return

    print(f"Backbone weights: {backbone_path}")
    print(f"Detection weights: {detection_path}")
    print(f"Mode: {args.mode}")

    # Load model
    if args.mode == "segmentation":
        model = load_backbone(backbone_path, device)
    else:
        model = load_combined_model(backbone_path, detection_path, device)

    # Create output dirs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Get test cases
    img_dir = DATA_DIR / "imagesTs"
    label_dir = DATA_DIR / "labelsTs"

    case_ids = sorted([
        p.stem.replace("_0000", "")
        for p in img_dir.glob("*_0000.nii.gz")
    ])
    if args.num_cases:
        case_ids = case_ids[:args.num_cases]
    print(f"\nProcessing {len(case_ids)} test cases...")

    all_results = []

    for case_id in case_ids:
        img_path = img_dir / f"{case_id}_0000.nii.gz"
        label_path = label_dir / f"{case_id}.nii.gz"

        print(f"\n--- {case_id} ---")
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        gt_boxes = load_gt_boxes(label_path)
        print(f"  Image shape: {img.shape}, GT boxes: {len(gt_boxes)}")

        # Run inference
        if args.mode == "segmentation":
            pred_boxes, pred_scores = segmentation_inference(
                model, img, device, threshold=args.threshold)
        else:
            pred_boxes, pred_scores = detection_inference(model, img, device)

        print(f"  Predictions: {len(pred_boxes)} boxes")
        if len(pred_scores) > 0:
            print(f"  Score range: {pred_scores.min():.3f} - {pred_scores.max():.3f}")

        # Save predictions
        pred_data = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': np.zeros(len(pred_boxes), dtype=int),
        }
        with open(OUT_DIR / f"{case_id}_boxes.pkl", 'wb') as f:
            pickle.dump(pred_data, f)

        all_results.append((case_id, pred_boxes, pred_scores, gt_boxes))

    # Evaluate
    print("\n" + "=" * 60)
    results = evaluate_predictions(all_results, iou_thresh=0.1)

    # Also evaluate with center-point matching
    print("\n--- Center-point matching (dist < 50 voxels) ---")
    total_tp_c, total_fp_c, total_fn_c = 0, 0, 0
    for case_id, pred_boxes, pred_scores, gt_boxes in all_results:
        matched_gt = set()
        tp = 0
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            order = np.argsort(-pred_scores)
            for pi in order:
                pc = np.array([(pred_boxes[pi][0]+pred_boxes[pi][2])/2,
                               (pred_boxes[pi][1]+pred_boxes[pi][3])/2,
                               (pred_boxes[pi][4]+pred_boxes[pi][5])/2])
                best_dist = float('inf')
                best_gi = -1
                for gi in range(len(gt_boxes)):
                    if gi in matched_gt:
                        continue
                    gc = np.array([(gt_boxes[gi][0]+gt_boxes[gi][2])/2,
                                   (gt_boxes[gi][1]+gt_boxes[gi][3])/2,
                                   (gt_boxes[gi][4]+gt_boxes[gi][5])/2])
                    dist = np.linalg.norm(pc - gc)
                    if dist < best_dist:
                        best_dist = dist
                        best_gi = gi
                if best_dist < 50 and best_gi >= 0:
                    matched_gt.add(best_gi)
                    tp += 1
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)
        total_tp_c += tp
        total_fp_c += fp
        total_fn_c += fn

    prec_c = total_tp_c / (total_tp_c + total_fp_c) if total_tp_c + total_fp_c > 0 else 0
    rec_c = total_tp_c / (total_tp_c + total_fn_c) if total_tp_c + total_fn_c > 0 else 0
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if prec_c + rec_c > 0 else 0
    print(f"TP={total_tp_c}, FP={total_fp_c}, FN={total_fn_c}")
    print(f"Precision={prec_c:.4f}, Recall={rec_c:.4f}, F1={f1_c:.4f}")

    # Save summary
    summary = {
        'mode': args.mode,
        'threshold': args.threshold,
        'iou_eval': results,
        'center_eval': {'tp': total_tp_c, 'fp': total_fp_c, 'fn': total_fn_c,
                        'precision': prec_c, 'recall': rec_c, 'f1': f1_c},
        'per_case': [(cid, len(pb), len(gb))
                     for cid, pb, _, gb in all_results],
    }
    with open(OUT_DIR / "ssl_abus_summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

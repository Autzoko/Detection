"""
Run YOLOv8 inference on 3D ABUS volumes (slice-by-slice) and reconstruct 3D bounding boxes.

Pipeline:
  1. Load 3D NIfTI volume
  2. Slice along z-axis → 2D images
  3. Run YOLOv8 detection on each slice
  4. Group consecutive slice detections with overlapping 2D boxes into 3D boxes
  5. Output 3D bounding boxes in nnDetection format for comparison

Usage:
    # Predict on test volumes
    python predict_and_reconstruct_3d.py \
        --model runs/detect/abus_lesion/weights/best.pt \
        --image_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted/imagesTs" \
        --output_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/yolo_predictions" \
        --conf 0.3

    # With ground truth evaluation
    python predict_and_reconstruct_3d.py \
        --model runs/detect/abus_lesion/weights/best.pt \
        --image_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted/imagesTs" \
        --label_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted/labelsTs" \
        --output_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/yolo_predictions" \
        --conf 0.3
"""

import argparse
import json
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 3D reconstruction from 2D slices")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLOv8 model")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with NIfTI test volumes (*_0000.nii.gz)")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Directory with ground truth instance labels (*.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold for YOLOv8 predictions")
    parser.add_argument("--iou_merge", type=float, default=0.3,
                        help="IoU threshold for merging 2D boxes across slices")
    parser.add_argument("--min_slices", type=int, default=3,
                        help="Minimum consecutive slices for a valid 3D box")
    parser.add_argument("--max_gap", type=int, default=2,
                        help="Maximum gap between slices to still merge")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLOv8 inference image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for slice inference")
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def iou_2d(box1, box2):
    """Compute IoU between two boxes [x_min, y_min, x_max, y_max]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def slice_volume_to_images(volume_data):
    """Convert 3D volume array to list of 2D images (along z/axis-0).

    Args:
        volume_data: numpy array of shape (z, y, x) or (C, z, y, x)

    Returns:
        list of 2D numpy arrays, each shape (y, x) as uint8
    """
    if volume_data.ndim == 4:
        # Take first channel
        volume_data = volume_data[0]

    slices = []
    for z in range(volume_data.shape[0]):
        s = volume_data[z].astype(np.float32)
        # Normalize to 0-255
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            s = (s - s_min) / (s_max - s_min) * 255
        else:
            s = np.zeros_like(s)
        # Convert to 3-channel uint8 for YOLO
        s = s.astype(np.uint8)
        s = np.stack([s, s, s], axis=-1)  # Grayscale → RGB
        slices.append(s)
    return slices


def predict_slices(model, slices, conf=0.3, imgsz=640, batch_size=32):
    """Run YOLOv8 on a list of 2D slice images.

    Returns:
        list of lists: per-slice detections, each detection is
            [x_min, y_min, x_max, y_max, confidence, class]
    """
    all_detections = []

    for batch_start in range(0, len(slices), batch_size):
        batch = slices[batch_start:batch_start + batch_size]
        results = model.predict(batch, conf=conf, imgsz=imgsz, verbose=False)

        for result in results:
            dets = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for box, c, cls in zip(boxes, confs, classes):
                    dets.append([float(box[0]), float(box[1]),
                                 float(box[2]), float(box[3]),
                                 float(c), int(cls)])
            all_detections.append(dets)

    return all_detections


def reconstruct_3d_boxes(per_slice_dets, iou_threshold=0.3, min_slices=3, max_gap=2):
    """Merge consecutive 2D detections into 3D bounding boxes.

    Args:
        per_slice_dets: list of lists, per_slice_dets[z] = [[x_min,y_min,x_max,y_max,conf,cls], ...]
        iou_threshold: IoU threshold for matching boxes across slices
        min_slices: minimum number of slices for a valid 3D detection
        max_gap: maximum gap (in slices) allowed between matched detections

    Returns:
        list of dicts, each with:
            'box_3d': [z_min, y_min, z_max, y_max, x_min, x_max] (nnDetection format)
            'confidence': float
            'n_slices': int
            'class': int
    """
    # Track active 3D box candidates
    # Each candidate: {2d_box_accum, z_slices, confidences, last_z, class}
    active_candidates = []
    finished_candidates = []

    for z_idx, dets in enumerate(per_slice_dets):
        matched_candidates = set()
        matched_dets = set()

        # Try to match each detection to an active candidate
        for det_idx, det in enumerate(dets):
            det_box = det[:4]
            det_conf = det[4]

            best_iou = 0
            best_cand_idx = -1

            for cand_idx, cand in enumerate(active_candidates):
                if cand_idx in matched_candidates:
                    continue
                # Compare with the candidate's latest 2D box
                cand_box = cand["latest_2d_box"]
                overlap = iou_2d(det_box, cand_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_cand_idx = cand_idx

            if best_iou >= iou_threshold and best_cand_idx >= 0:
                # Extend existing candidate
                cand = active_candidates[best_cand_idx]
                cand["z_slices"].append(z_idx)
                cand["confidences"].append(det_conf)
                cand["latest_2d_box"] = det_box
                # Accumulate 2D box bounds
                cand["x_min"] = min(cand["x_min"], det_box[0])
                cand["y_min"] = min(cand["y_min"], det_box[1])
                cand["x_max"] = max(cand["x_max"], det_box[2])
                cand["y_max"] = max(cand["y_max"], det_box[3])
                cand["last_z"] = z_idx
                matched_candidates.add(best_cand_idx)
                matched_dets.add(det_idx)

        # Start new candidates for unmatched detections
        for det_idx, det in enumerate(dets):
            if det_idx in matched_dets:
                continue
            active_candidates.append({
                "z_slices": [z_idx],
                "confidences": [det[4]],
                "latest_2d_box": det[:4],
                "x_min": det[0],
                "y_min": det[1],
                "x_max": det[2],
                "y_max": det[3],
                "last_z": z_idx,
                "class": det[5],
            })

        # Check for expired candidates (gap too large)
        still_active = []
        for cand in active_candidates:
            if z_idx - cand["last_z"] > max_gap:
                finished_candidates.append(cand)
            else:
                still_active.append(cand)
        active_candidates = still_active

    # Finalize remaining active candidates
    finished_candidates.extend(active_candidates)

    # Filter by minimum slices and build 3D boxes
    boxes_3d = []
    for cand in finished_candidates:
        n_slices = len(cand["z_slices"])
        if n_slices < min_slices:
            continue

        z_min = min(cand["z_slices"])
        z_max = max(cand["z_slices"])
        mean_conf = float(np.mean(cand["confidences"]))
        max_conf = float(np.max(cand["confidences"]))

        # nnDetection box format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
        # where d0=z, d1=y, d2=x
        box_3d = [
            float(z_min),           # d0_min (z)
            float(cand["y_min"]),   # d1_min (y)
            float(z_max),           # d0_max (z)
            float(cand["y_max"]),   # d1_max (y)
            float(cand["x_min"]),   # d2_min (x)
            float(cand["x_max"]),   # d2_max (x)
        ]

        boxes_3d.append({
            "box_3d": box_3d,
            "confidence": max_conf,
            "mean_confidence": mean_conf,
            "n_slices": n_slices,
            "z_range": [z_min, z_max],
            "class": cand["class"],
        })

    # Sort by confidence
    boxes_3d.sort(key=lambda x: x["confidence"], reverse=True)
    return boxes_3d


def load_gt_boxes(label_path):
    """Load ground truth 3D boxes from instance segmentation NIfTI.

    Returns list of boxes in nnDetection format [z_min, y_min, z_max, y_max, x_min, x_max].
    """
    import SimpleITK as sitk
    from skimage.measure import regionprops

    label_sitk = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_sitk)  # (z, y, x)

    if label_arr.max() == 0:
        return []

    gt_boxes = []
    for prop in regionprops(label_arr.astype(np.int32)):
        bbox = prop.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        # Convert to nnDetection format: [d0_min, d1_min, d0_max, d1_max, d2_min, d2_max]
        gt_box = [bbox[0], bbox[1], bbox[3], bbox[4], bbox[2], bbox[5]]
        gt_boxes.append(gt_box)

    return gt_boxes


def iou_3d(box1, box2):
    """Compute 3D IoU. Boxes in [z_min, y_min, z_max, y_max, x_min, x_max] format."""
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

    return inter / union if union > 0 else 0


def evaluate_predictions(all_predictions, all_gt_boxes, iou_thresholds=(0.1, 0.2, 0.3, 0.5)):
    """Evaluate 3D predictions against ground truth.

    Args:
        all_predictions: dict of case_id -> list of prediction dicts
        all_gt_boxes: dict of case_id -> list of GT boxes
        iou_thresholds: IoU thresholds for matching

    Returns:
        dict with evaluation metrics
    """
    results = {}

    for iou_thresh in iou_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0

        for case_id in all_gt_boxes:
            gt_boxes = all_gt_boxes[case_id]
            preds = all_predictions.get(case_id, [])
            total_gt += len(gt_boxes)

            gt_matched = [False] * len(gt_boxes)

            for pred in preds:
                pred_box = pred["box_3d"]
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    overlap = iou_3d(pred_box, gt_box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh:
                    total_tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    total_fp += 1

            total_fn += sum(1 for m in gt_matched if not m)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        n_cases = len(all_gt_boxes)
        fp_per_case = total_fp / n_cases if n_cases > 0 else 0

        results[f"IoU={iou_thresh}"] = {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "FP/case": round(fp_per_case, 2),
        }

    return results


def main():
    args = parse_args()

    import SimpleITK as sitk
    from ultralytics import YOLO

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Find test volumes
    image_dir = Path(args.image_dir)
    volume_files = sorted(image_dir.glob("*_0000.nii.gz"))
    print(f"Found {len(volume_files)} test volumes")

    # Load GT labels if provided
    has_gt = args.label_dir is not None
    label_dir = Path(args.label_dir) if has_gt else None

    all_predictions = {}
    all_gt_boxes = {}

    for vol_idx, vol_path in enumerate(volume_files):
        case_id = vol_path.name.replace("_0000.nii.gz", "")
        print(f"\n[{vol_idx+1}/{len(volume_files)}] Processing {case_id}...")

        # Load volume
        vol_sitk = sitk.ReadImage(str(vol_path))
        vol_arr = sitk.GetArrayFromImage(vol_sitk)  # (z, y, x)
        print(f"  Volume shape: {vol_arr.shape}")

        # Slice into 2D images
        slices = slice_volume_to_images(vol_arr)
        print(f"  Generated {len(slices)} slices")

        # Run YOLO prediction
        per_slice_dets = predict_slices(
            model, slices,
            conf=args.conf,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
        )
        n_dets = sum(len(d) for d in per_slice_dets)
        n_pos_slices = sum(1 for d in per_slice_dets if len(d) > 0)
        print(f"  2D detections: {n_dets} across {n_pos_slices} slices")

        # Reconstruct 3D boxes
        boxes_3d = reconstruct_3d_boxes(
            per_slice_dets,
            iou_threshold=args.iou_merge,
            min_slices=args.min_slices,
            max_gap=args.max_gap,
        )
        print(f"  3D boxes reconstructed: {len(boxes_3d)}")

        all_predictions[case_id] = boxes_3d

        # Save per-case predictions (nnDetection-compatible format)
        pred_boxes = np.array([b["box_3d"] for b in boxes_3d]) if boxes_3d else np.zeros((0, 6))
        pred_scores = np.array([b["confidence"] for b in boxes_3d]) if boxes_3d else np.zeros(0)
        pred_labels = np.zeros(len(boxes_3d), dtype=np.int64)

        case_pred = {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
        }
        with open(output_dir / f"{case_id}_boxes.pkl", "wb") as f:
            pickle.dump(case_pred, f)

        # Load GT if available
        if has_gt:
            gt_path = label_dir / f"{case_id}.nii.gz"
            if gt_path.exists():
                gt_boxes = load_gt_boxes(gt_path)
                all_gt_boxes[case_id] = gt_boxes
                print(f"  GT boxes: {len(gt_boxes)}")

    # Evaluate if GT available
    if has_gt and all_gt_boxes:
        print("\n" + "=" * 60)
        print("=== 3D Detection Evaluation ===")
        print("=" * 60)

        eval_results = evaluate_predictions(all_predictions, all_gt_boxes)

        for thresh_name, metrics in eval_results.items():
            print(f"\n  {thresh_name}:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

        # Save evaluation results
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

    # Save summary
    summary = {
        "model": args.model,
        "conf_threshold": args.conf,
        "iou_merge": args.iou_merge,
        "min_slices": args.min_slices,
        "max_gap": args.max_gap,
        "n_volumes": len(volume_files),
        "per_case": {
            case_id: {
                "n_predictions": len(preds),
                "max_conf": max((p["confidence"] for p in preds), default=0),
            }
            for case_id, preds in all_predictions.items()
        },
    }
    with open(output_dir / "prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_preds = sum(len(p) for p in all_predictions.values())
    print(f"\n=== Summary ===")
    print(f"  Total 3D predictions: {total_preds} across {len(volume_files)} volumes")
    print(f"  Avg predictions/volume: {total_preds/len(volume_files):.1f}")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

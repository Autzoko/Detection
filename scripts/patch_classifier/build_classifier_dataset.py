"""
Build training dataset for 3D patch classifier using nnDetection training volumes.

Positives: GT bounding boxes (+ jittered copies for augmentation)
Negatives: Random regions from each volume that don't overlap any GT box

No nnDetection predictions needed — works directly from volumes + GT masks.

Usage:
    python build_classifier_dataset.py --config config.yaml
    python build_classifier_dataset.py --config config.yaml --dry_run
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml
from scipy.ndimage import zoom
from skimage.measure import regionprops


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def extract_gt_boxes(label_path):
    """Extract GT bboxes from instance segmentation NIfTI."""
    label_sitk = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_sitk).astype(np.int32)  # (z, y, x)
    if label_arr.max() == 0:
        return []
    gt_boxes = []
    for prop in regionprops(label_arr):
        bbox = prop.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        # nnDetection format: [z_min, y_min, z_max, y_max, x_min, x_max]
        gt_boxes.append([bbox[0], bbox[1], bbox[3], bbox[4], bbox[2], bbox[5]])
    return gt_boxes


def crop_and_resize_patch(volume, box, padding_fraction, target_size):
    """Crop 3D patch centered on box with padding, resize to target_size."""
    z_min, y_min, z_max, y_max, x_min, x_max = box
    dz, dy, dx = z_max - z_min, y_max - y_min, x_max - x_min
    pad_z = int(dz * padding_fraction)
    pad_y = int(dy * padding_fraction)
    pad_x = int(dx * padding_fraction)

    cz0, cz1 = int(z_min) - pad_z, int(z_max) + pad_z
    cy0, cy1 = int(y_min) - pad_y, int(y_max) + pad_y
    cx0, cx1 = int(x_min) - pad_x, int(x_max) + pad_x

    # Ensure minimum size
    for _ in range(3):
        if cz1 - cz0 < 4: cz0 -= 1; cz1 += 1
        if cy1 - cy0 < 4: cy0 -= 1; cy1 += 1
        if cx1 - cx0 < 4: cx0 -= 1; cx1 += 1

    vz, vy, vx = volume.shape
    pad_before = [max(0, -cz0), max(0, -cy0), max(0, -cx0)]
    pad_after = [max(0, cz1 - vz), max(0, cy1 - vy), max(0, cx1 - vx)]

    patch = volume[max(0, cz0):min(vz, cz1), max(0, cy0):min(vy, cy1), max(0, cx0):min(vx, cx1)]
    if any(p > 0 for p in pad_before + pad_after):
        patch = np.pad(patch, list(zip(pad_before, pad_after)), mode="constant", constant_values=0)

    if 0 in patch.shape:
        return np.zeros(target_size, dtype=np.float32)

    zoom_factors = [t / s for t, s in zip(target_size, patch.shape)]
    patch_resized = zoom(patch.astype(np.float32), zoom_factors, order=1)

    if patch_resized.shape != tuple(target_size):
        result = np.zeros(target_size, dtype=np.float32)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(patch_resized.shape, target_size))
        result[slices] = patch_resized[slices]
        return result
    return patch_resized


def jitter_box(box, volume_shape, jitter_frac=0.15):
    """Create a jittered copy of a box by shifting center and scaling slightly."""
    z_min, y_min, z_max, y_max, x_min, x_max = box
    dz, dy, dx = z_max - z_min, y_max - y_min, x_max - x_min
    cz, cy, cx = (z_min + z_max) / 2, (y_min + y_max) / 2, (x_min + x_max) / 2

    # Random center shift
    cz += random.uniform(-jitter_frac, jitter_frac) * dz
    cy += random.uniform(-jitter_frac, jitter_frac) * dy
    cx += random.uniform(-jitter_frac, jitter_frac) * dx

    # Random scale
    scale = random.uniform(1.0 - jitter_frac, 1.0 + jitter_frac)
    dz *= scale
    dy *= scale
    dx *= scale

    vz, vy, vx = volume_shape
    new_box = [
        max(0, cz - dz / 2), max(0, cy - dy / 2),
        min(vz, cz + dz / 2), min(vy, cy + dy / 2),
        max(0, cx - dx / 2), min(vx, cx + dx / 2),
    ]
    return new_box


def sample_random_negative(volume_shape, gt_boxes, box_size_range, max_attempts=50):
    """Sample a random bbox that doesn't overlap with any GT box (IoU < 0.05)."""
    vz, vy, vx = volume_shape
    for _ in range(max_attempts):
        dz = random.randint(box_size_range[0][0], box_size_range[0][1])
        dy = random.randint(box_size_range[1][0], box_size_range[1][1])
        dx = random.randint(box_size_range[2][0], box_size_range[2][1])

        z0 = random.randint(0, max(0, vz - dz))
        y0 = random.randint(0, max(0, vy - dy))
        x0 = random.randint(0, max(0, vx - dx))

        box = [z0, y0, z0 + dz, y0 + dy, x0, x0 + dx]
        overlaps = [iou_3d(box, gt) for gt in gt_boxes] if gt_boxes else [0.0]
        if max(overlaps) < 0.05:
            return box
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print statistics without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    ds_cfg = cfg["dataset"]

    images_dir = Path(paths["train_images_dir"])
    labels_dir = Path(paths["train_labels_dir"])
    dataset_dir = Path(paths["dataset_dir"])

    target_size = ds_cfg["patch_size"]
    padding_frac = ds_cfg["padding_fraction"]
    p_low = ds_cfg["percentile_low"]
    p_high = ds_cfg["percentile_high"]
    n_jitter = ds_cfg.get("jittered_pos_per_gt", 3)
    n_neg_per_vol = ds_cfg.get("random_neg_per_volume", 10)

    # Find all training volumes
    vol_files = sorted(images_dir.glob("*_0000.nii.gz"))
    case_ids = [f.name.replace("_0000.nii.gz", "") for f in vol_files]
    print(f"Found {len(case_ids)} training volumes")

    # Volume-level split: all for training, small subset also for val
    random.seed(ds_cfg["seed"])
    shuffled = list(case_ids)
    random.shuffle(shuffled)
    val_fraction = ds_cfg.get("val_fraction", 0.15)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_cases = set(shuffled[:n_val])
    train_cases = set(case_ids)

    print(f"Training on ALL {len(train_cases)} cases")
    print(f"Val subset ({len(val_cases)} cases) for early stopping only")

    if not args.dry_run:
        for split in ["train", "val"]:
            (dataset_dir / split / "patches").mkdir(parents=True, exist_ok=True)

    stats = {
        "train": {"pos": 0, "neg": 0, "gt_total": 0, "volumes": 0},
        "val": {"pos": 0, "neg": 0, "gt_total": 0, "volumes": 0},
    }
    all_samples = {"train": [], "val": []}

    for case_idx, case_id in enumerate(case_ids):
        splits = ["train"]
        if case_id in val_cases:
            splits.append("val")
        split_label = "train+val" if case_id in val_cases else "train"
        print(f"\n[{case_idx+1}/{len(case_ids)}] {case_id} ({split_label})")

        # Load volume
        vol_path = images_dir / f"{case_id}_0000.nii.gz"
        vol_sitk = sitk.ReadImage(str(vol_path))
        volume = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)
        vol_shape = volume.shape

        # Volume-level percentiles
        v_low = np.percentile(volume, p_low)
        v_high = np.percentile(volume, p_high)

        def normalize(patch):
            patch = np.clip(patch, v_low, v_high)
            if v_high > v_low:
                return ((patch - v_low) / (v_high - v_low)).astype(np.float32)
            return np.zeros_like(patch, dtype=np.float32)

        # Load GT
        gt_path = labels_dir / f"{case_id}.nii.gz"
        gt_boxes = extract_gt_boxes(gt_path) if gt_path.exists() else []
        print(f"  GT boxes: {len(gt_boxes)}")

        for split in splits:
            stats[split]["gt_total"] += len(gt_boxes)
            stats[split]["volumes"] += 1

        # === POSITIVES: GT boxes + jittered copies ===
        for g_idx, gt_box in enumerate(gt_boxes):
            # Original GT box
            patch = crop_and_resize_patch(volume, gt_box, padding_frac, target_size)
            patch = normalize(patch)
            sample_name = f"{case_id}_gt{g_idx:04d}"
            sample_info = {
                "name": sample_name,
                "case_id": case_id,
                "box": gt_box,
                "label": 1,
                "source": "gt",
            }

            for split in splits:
                if not args.dry_run:
                    np.save(dataset_dir / split / "patches" / f"{sample_name}.npy", patch)
                all_samples[split].append(sample_info)
                stats[split]["pos"] += 1

            # Jittered copies
            for j_idx in range(n_jitter):
                jbox = jitter_box(gt_box, vol_shape)
                patch = crop_and_resize_patch(volume, jbox, padding_frac, target_size)
                patch = normalize(patch)
                sample_name = f"{case_id}_gt{g_idx:04d}_j{j_idx}"
                sample_info = {
                    "name": sample_name,
                    "case_id": case_id,
                    "box": jbox,
                    "label": 1,
                    "source": "gt_jittered",
                }

                for split in splits:
                    if not args.dry_run:
                        np.save(dataset_dir / split / "patches" / f"{sample_name}.npy", patch)
                    all_samples[split].append(sample_info)
                    stats[split]["pos"] += 1

        # === NEGATIVES: random regions ===
        if gt_boxes:
            # Use GT box sizes as reference for random negatives
            gt_sizes_z = [b[2] - b[0] for b in gt_boxes]
            gt_sizes_y = [b[3] - b[1] for b in gt_boxes]
            gt_sizes_x = [b[5] - b[4] for b in gt_boxes]
            size_range = [
                [max(10, int(min(gt_sizes_z) * 0.5)), int(max(gt_sizes_z) * 1.5)],
                [max(5, int(min(gt_sizes_y) * 0.5)), int(max(gt_sizes_y) * 1.5)],
                [max(10, int(min(gt_sizes_x) * 0.5)), int(max(gt_sizes_x) * 1.5)],
            ]
        else:
            size_range = [[20, 100], [10, 50], [30, 150]]

        for r_idx in range(n_neg_per_vol):
            rand_box = sample_random_negative(vol_shape, gt_boxes, size_range)
            if rand_box is None:
                continue

            patch = crop_and_resize_patch(volume, rand_box, padding_frac, target_size)
            patch = normalize(patch)
            sample_name = f"{case_id}_neg{r_idx:04d}"
            sample_info = {
                "name": sample_name,
                "case_id": case_id,
                "box": rand_box,
                "label": 0,
                "source": "random_negative",
            }

            for split in splits:
                if not args.dry_run:
                    np.save(dataset_dir / split / "patches" / f"{sample_name}.npy", patch)
                all_samples[split].append(sample_info)
                stats[split]["neg"] += 1

    # Balance classes
    print("\n=== Balancing Classes ===")
    for split in ["train", "val"]:
        samples = all_samples[split]
        pos_samples = [s for s in samples if s["label"] == 1]
        neg_samples = [s for s in samples if s["label"] == 0]

        print(f"\n  {split}: {len(pos_samples)} pos, {len(neg_samples)} neg")

        # Oversample positives if heavily imbalanced
        if ds_cfg.get("oversample_tp", False) and len(pos_samples) > 0:
            if len(neg_samples) > len(pos_samples) * 2:
                repeat = max(1, len(neg_samples) // (len(pos_samples) * 2))
                pos_samples = pos_samples * repeat
                print(f"  Oversampled positives by {repeat}x → {len(pos_samples)}")

        balanced = pos_samples + neg_samples
        random.shuffle(balanced)
        all_samples[split] = balanced

        print(f"  Final: {len(pos_samples)} pos, {len(neg_samples)} neg = {len(balanced)} total")

    # Save manifests
    if not args.dry_run:
        for split in ["train", "val"]:
            manifest_path = dataset_dir / split / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(all_samples[split], f, indent=2)
            print(f"\n  Saved {split} manifest: {manifest_path}")

        split_info = {
            "train_cases": sorted(train_cases),
            "val_cases": sorted(val_cases),
            "note": "All cases used for training; val is a subset for early stopping only",
            "stats": stats,
            "config": ds_cfg,
        }
        with open(dataset_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("=== Dataset Build Summary ===")
    print("=" * 60)
    for split in ["train", "val"]:
        s = stats[split]
        n_final = len(all_samples[split])
        n_pos = sum(1 for x in all_samples[split] if x["label"] == 1)
        print(f"\n  {split}:")
        print(f"    Volumes:    {s['volumes']}")
        print(f"    GT lesions: {s['gt_total']}")
        print(f"    Positives:  {s['pos']} (GT + jittered)")
        print(f"    Negatives:  {s['neg']}")
        print(f"    Final:      {n_final} ({n_pos} pos, {n_final - n_pos} neg)")


if __name__ == "__main__":
    main()

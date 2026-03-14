"""
Prepare 3D Breast ABUS data for nnDetection (Task100_BreastABUS).

Data layout on disk
-------------------
  /Volumes/Autzoko/Dataset/third-party/data/
    2类/  3类/  4类/          – BI-RADS class folders (training candidates)
    度影AI数据/                – raw .ai volumes, NO annotations → skipped
    已标注及BI-rads分类20260123/  – test set

Images  : .nii  (NIfTI, uint8 ultrasound)
Labels  : _nii_Label.tar  →  JSON with BoundingBoxLabelModel (2-D BBs on slices)

This script:
  1. Discovers image/label pairs across all class folders + test set
  2. Deduplicates by MD5 hash (prefer test-set copy, then annotated)
  3. Converts 2-D BB annotations → 3-D ellipsoid instance-segmentation masks
  4. Splits: test-set → test; 15 % of rest (stratified by class) → val; rest → train
  5. Writes nnDetection raw_splitted/ layout + dataset.json + dataset_statistics.csv
"""

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/Volumes/Autzoko/Dataset/third-party/data")
CLASS_FOLDERS = ["2类", "3类", "4类"]
TEST_FOLDER = "已标注及BI-rads分类20260123"
TASK_NAME = "Task100_BreastABUS"

BIRADS_MAP = {"2类": "birads2", "3类": "birads3", "4类": "birads4"}


# ---------------------------------------------------------------------------
# Helpers – file discovery
# ---------------------------------------------------------------------------


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return hex MD5 digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_pairs(root: Path) -> list[dict]:
    """Recursively find .nii images that have a matching _nii_Label.tar."""
    pairs = []
    for nii in sorted(root.rglob("*.nii")):
        stem = nii.stem  # e.g. 20230413093015227123456
        tar_name = f"{stem}_nii_Label.tar"
        tar_path = nii.parent / tar_name
        if tar_path.is_file():
            pairs.append({"image": nii, "label_tar": tar_path})
    return pairs


# ---------------------------------------------------------------------------
# Helpers – annotation parsing
# ---------------------------------------------------------------------------


def parse_label_tar(tar_path: Path) -> dict:
    """Extract and parse the JSON from a _nii_Label.tar file."""
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                f = tf.extractfile(member)
                if f is not None:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    raise ValueError(f"No JSON found in {tar_path}")


def extract_bboxes(label_json: dict) -> list[dict]:
    """
    Return list of per-lesion 3-D bounding boxes in *physical* coordinates.

    Each BB entry from the annotator is a 2-D rectangle on one slice-type:
      SliceType 0 → sagittal (X fixed), gives Y/Z ranges
      SliceType 1 → coronal  (Y fixed), gives X/Z ranges
      SliceType 2 → axial    (Z fixed), gives X/Y ranges

    We group by Label and merge across views to get the 3-D extent.
    If only one view is available for a dimension, we estimate using
    depth = min(w, h) * 0.8.
    """
    models = label_json.get("Models", {})
    bb_entries = models.get("BoundingBoxLabelModel") or []

    if not bb_entries:
        return []

    # Group by label
    by_label: dict[int, list[dict]] = defaultdict(list)
    for bb in bb_entries:
        by_label[bb["Label"]].append(bb)

    results = []
    for label_id, bbs in sorted(by_label.items()):
        x_ranges, y_ranges, z_ranges = [], [], []

        for bb in bbs:
            p1 = bb["p1"]
            p2 = bb["p2"]
            st = bb["SliceType"]

            if st == 0:  # sagittal – X fixed
                y_ranges.append((min(p1[1], p2[1]), max(p1[1], p2[1])))
                z_ranges.append((min(p1[2], p2[2]), max(p1[2], p2[2])))
            elif st == 1:  # coronal – Y fixed
                x_ranges.append((min(p1[0], p2[0]), max(p1[0], p2[0])))
                z_ranges.append((min(p1[2], p2[2]), max(p1[2], p2[2])))
            elif st == 2:  # axial – Z fixed
                x_ranges.append((min(p1[0], p2[0]), max(p1[0], p2[0])))
                y_ranges.append((min(p1[1], p2[1]), max(p1[1], p2[1])))

        def merge_ranges(ranges):
            if not ranges:
                return None
            lo = min(r[0] for r in ranges)
            hi = max(r[1] for r in ranges)
            return (lo, hi)

        xr = merge_ranges(x_ranges)
        yr = merge_ranges(y_ranges)
        zr = merge_ranges(z_ranges)

        # Estimate missing dimension
        known_sizes = []
        if xr:
            known_sizes.append(xr[1] - xr[0])
        if yr:
            known_sizes.append(yr[1] - yr[0])
        if zr:
            known_sizes.append(zr[1] - zr[0])

        if not known_sizes:
            continue  # degenerate

        est_depth = min(known_sizes) * 0.8

        if xr is None:
            # Need a center – use midpoint of available FrameCount * spacing
            # But we don't know X center precisely; use average of Y/Z centers
            # Actually we need a center X. Let's get it from a sagittal BB frame.
            cx = None
            for bb in bbs:
                if bb["SliceType"] == 0:
                    cx = bb["p1"][0]  # fixed X in physical coords
                    break
            if cx is None:
                # Use midpoints from other views
                if y_ranges or z_ranges:
                    # Approximate: pick the X from the first available BB
                    cx = bbs[0]["p1"][0]
            if cx is not None:
                xr = (cx - est_depth / 2, cx + est_depth / 2)
            else:
                continue

        if yr is None:
            cy = None
            for bb in bbs:
                if bb["SliceType"] == 1:
                    cy = bb["p1"][1]
                    break
            if cy is None:
                cy = bbs[0]["p1"][1]
            if cy is not None:
                yr = (cy - est_depth / 2, cy + est_depth / 2)
            else:
                continue

        if zr is None:
            cz = None
            for bb in bbs:
                if bb["SliceType"] == 2:
                    cz = bb["p1"][2]
                    break
            if cz is None:
                cz = bbs[0]["p1"][2]
            if cz is not None:
                zr = (cz - est_depth / 2, cz + est_depth / 2)
            else:
                continue

        results.append({
            "label_id": label_id,
            "x_range_phys": xr,
            "y_range_phys": yr,
            "z_range_phys": zr,
        })

    return results


def create_instance_mask(image_itk, bboxes: list[dict]) -> tuple:
    """
    Create an instance segmentation mask from 3-D bounding boxes.

    Each lesion is filled with an ellipsoid inside its bounding box.
    Returns (mask_itk, instance_classes_dict).
    """
    spacing = image_itk.GetSpacing()  # (sx, sy, sz)
    size = image_itk.GetSize()  # (nx, ny, nz)

    mask = np.zeros((size[2], size[1], size[0]), dtype=np.float32)  # z, y, x
    instance_classes = {}

    for inst_id, bb in enumerate(bboxes, start=1):
        # Convert physical coordinates to voxel indices
        x0 = max(0, int(np.floor(bb["x_range_phys"][0] / spacing[0])))
        x1 = min(size[0] - 1, int(np.ceil(bb["x_range_phys"][1] / spacing[0])))
        y0 = max(0, int(np.floor(bb["y_range_phys"][0] / spacing[1])))
        y1 = min(size[1] - 1, int(np.ceil(bb["y_range_phys"][1] / spacing[1])))
        z0 = max(0, int(np.floor(bb["z_range_phys"][0] / spacing[2])))
        z1 = min(size[2] - 1, int(np.ceil(bb["z_range_phys"][1] / spacing[2])))

        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            logger.warning(f"Degenerate BB for instance {inst_id}, skipping")
            continue

        # Create ellipsoid inside the bounding box
        cz = (z0 + z1) / 2.0
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        rz = (z1 - z0) / 2.0
        ry = (y1 - y0) / 2.0
        rx = (x1 - x0) / 2.0

        # Fill ellipsoid – iterate only over the bounding box region
        zz, yy, xx = np.mgrid[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        ellipsoid = (
            ((zz - cz) / rz) ** 2
            + ((yy - cy) / ry) ** 2
            + ((xx - cx) / rx) ** 2
        ) <= 1.0

        # Only write where mask is still 0 (no overlap with earlier instances)
        region = mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        region[ellipsoid & (region == 0)] = inst_id

        # All lesions → class 0 ("Lesion") for detection
        instance_classes[str(inst_id)] = 0

    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.SetSpacing(spacing)
    mask_itk.SetOrigin(image_itk.GetOrigin())
    mask_itk.SetDirection(image_itk.GetDirection())

    return mask_itk, instance_classes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def discover_all_volumes(data_root: Path) -> list[dict]:
    """Find all annotated image/label pairs across class folders and test set."""
    volumes = []

    # Training-candidate folders
    for cls_folder in CLASS_FOLDERS:
        folder = data_root / cls_folder
        if not folder.is_dir():
            logger.warning(f"Class folder not found: {folder}")
            continue
        pairs = find_pairs(folder)
        logger.info(f"{cls_folder}: found {len(pairs)} annotated volumes")
        for p in pairs:
            p["source_folder"] = cls_folder
            p["lesion_class"] = BIRADS_MAP[cls_folder]
            p["is_test"] = False
        volumes.extend(pairs)

    # Test set
    test_root = data_root / TEST_FOLDER
    if test_root.is_dir():
        pairs = find_pairs(test_root)
        logger.info(f"Test set: found {len(pairs)} annotated volumes")
        for p in pairs:
            p["source_folder"] = TEST_FOLDER
            p["lesion_class"] = "test_mixed"
            p["is_test"] = True
        volumes.extend(pairs)
    else:
        logger.warning(f"Test folder not found: {test_root}")

    return volumes


def deduplicate(volumes: list[dict]) -> list[dict]:
    """
    Deduplicate volumes by MD5 hash of the .nii image file.
    Preference: test-set copy > annotated copy from higher-class folder.
    """
    logger.info(f"Computing MD5 hashes for {len(volumes)} volumes...")

    # Group by file size first for efficiency
    by_size: dict[int, list[dict]] = defaultdict(list)
    for v in volumes:
        sz = v["image"].stat().st_size
        v["file_size"] = sz
        by_size[sz].append(v)

    # Only compute MD5 for volumes with matching file sizes
    for v in volumes:
        v["md5"] = None  # placeholder

    for sz, group in by_size.items():
        if len(group) == 1:
            group[0]["md5"] = f"unique_{sz}"  # skip expensive hash
        else:
            for v in group:
                v["md5"] = md5_file(v["image"])
                logger.debug(f"  MD5 {v['image'].name}: {v['md5']}")

    # Deduplicate: keep best copy per hash
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for v in volumes:
        by_hash[v["md5"]].append(v)

    unique = []
    duplicates = []
    for md5, group in by_hash.items():
        if len(group) == 1:
            group[0]["is_duplicate"] = False
            unique.append(group[0])
        else:
            # Sort: test copies first, then by class folder
            group.sort(key=lambda v: (not v["is_test"], v["source_folder"]))
            group[0]["is_duplicate"] = False
            unique.append(group[0])
            for dup in group[1:]:
                dup["is_duplicate"] = True
                duplicates.append(dup)

    logger.info(
        f"Deduplication: {len(volumes)} → {len(unique)} unique "
        f"({len(duplicates)} duplicates removed)"
    )
    return unique, duplicates


def assign_splits(
    volumes: list[dict], val_fraction: float = 0.15, seed: int = 42
) -> list[dict]:
    """Assign train/val/test splits. Stratify val by source class folder."""
    for v in volumes:
        if v["is_test"]:
            v["split"] = "test"

    trainval = [v for v in volumes if not v["is_test"]]

    if not trainval:
        return volumes

    classes = [v["lesion_class"] for v in trainval]
    indices = list(range(len(trainval)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=classes,
    )

    for i in train_idx:
        trainval[i]["split"] = "train"
    for i in val_idx:
        trainval[i]["split"] = "val"

    splits = defaultdict(int)
    for v in volumes:
        splits[v["split"]] += 1
    logger.info(f"Splits: {dict(splits)}")

    return volumes


def process_volume(vol: dict, case_id: str, img_dir: Path, lbl_dir: Path) -> dict:
    """
    Process one volume: read image, parse annotations, create mask, save.
    Returns stats dict.
    """
    # Read image
    image_itk = sitk.ReadImage(str(vol["image"]))
    size = image_itk.GetSize()
    spacing = image_itk.GetSpacing()

    # Parse annotations
    try:
        label_json = parse_label_tar(vol["label_tar"])
    except Exception as e:
        logger.error(f"Failed to parse label tar {vol['label_tar']}: {e}")
        return None

    bboxes = extract_bboxes(label_json)
    if not bboxes:
        logger.warning(f"No bounding boxes found for {vol['image'].name}")
        return None

    # Create instance mask
    mask_itk, instance_classes = create_instance_mask(image_itk, bboxes)

    if not instance_classes:
        logger.warning(f"No valid instances for {vol['image'].name}")
        return None

    # Save image (single modality → _0000)
    sitk.WriteImage(image_itk, str(img_dir / f"{case_id}_0000.nii.gz"))

    # Save mask
    sitk.WriteImage(mask_itk, str(lbl_dir / f"{case_id}.nii.gz"))

    # Save instance metadata JSON
    meta = {"instances": instance_classes}
    with open(lbl_dir / f"{case_id}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "volume_id": case_id,
        "source_folder": vol["source_folder"],
        "image_path": str(vol["image"]),
        "mask_path": str(vol["label_tar"]),
        "shape": f"{size[0]}x{size[1]}x{size[2]}",
        "spacing": f"{spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}",
        "num_lesions": len(instance_classes),
        "lesion_class": vol["lesion_class"],
        "split": vol["split"],
        "is_duplicate": vol.get("is_duplicate", False),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Breast ABUS data for nnDetection"
    )
    parser.add_argument(
        "--det_data",
        type=str,
        default=os.environ.get("det_data", ""),
        help="nnDetection data root (default: $det_data env var)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DATA_ROOT),
        help="Raw data root directory",
    )
    args = parser.parse_args()

    if not args.det_data:
        raise RuntimeError(
            "Set det_data env var or pass --det_data /path/to/det_data"
        )

    det_data = Path(args.det_data)
    data_root = Path(args.data_root)

    # ---- 1. Discover all annotated volumes ----
    logger.info("Discovering annotated volumes...")
    volumes = discover_all_volumes(data_root)
    logger.info(f"Total annotated volumes found: {len(volumes)}")

    if not volumes:
        logger.error("No annotated volumes found. Exiting.")
        return

    # ---- 2. Deduplicate ----
    unique_volumes, duplicates = deduplicate(volumes)

    # ---- 3. Assign splits ----
    unique_volumes = assign_splits(unique_volumes)

    # ---- 4. Create output directory structure ----
    task_dir = det_data / TASK_NAME
    raw_dir = task_dir / "raw_splitted"

    dirs = {}
    for split_name, img_sub, lbl_sub in [
        ("train", "imagesTr", "labelsTr"),
        ("val", "imagesTr", "labelsTr"),  # val goes into Tr; nnDetection splits internally
        ("test", "imagesTs", "labelsTs"),
    ]:
        dirs[f"{split_name}_img"] = raw_dir / img_sub
        dirs[f"{split_name}_lbl"] = raw_dir / lbl_sub

    for d in set(dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 5. Process and convert each volume ----
    stats_rows = []
    case_counter = 0

    for vol in sorted(unique_volumes, key=lambda v: (v["split"], v["source_folder"], str(v["image"]))):
        split = vol["split"]
        if split in ("train", "val"):
            img_dir = dirs["train_img"]
            lbl_dir = dirs["train_lbl"]
        else:
            img_dir = dirs["test_img"]
            lbl_dir = dirs["test_lbl"]

        case_id = f"case_{case_counter:05d}"
        case_counter += 1

        logger.info(
            f"[{case_counter}/{len(unique_volumes)}] {split}: "
            f"{vol['image'].name} → {case_id}"
        )

        row = process_volume(vol, case_id, img_dir, lbl_dir)
        if row is not None:
            stats_rows.append(row)
        else:
            logger.warning(f"Skipped {vol['image'].name} (no valid annotations)")

    # Also record duplicates in stats
    for dup in duplicates:
        stats_rows.append({
            "volume_id": "DUPLICATE",
            "source_folder": dup["source_folder"],
            "image_path": str(dup["image"]),
            "mask_path": str(dup["label_tar"]),
            "shape": "",
            "spacing": "",
            "num_lesions": "",
            "lesion_class": dup["lesion_class"],
            "split": "excluded",
            "is_duplicate": True,
        })

    # ---- 6. Write dataset.json ----
    dataset_meta = {
        "task": TASK_NAME,
        "name": "Breast ABUS Lesion Detection",
        "target_class": 0,
        "test_labels": True,
        "labels": {"0": "Lesion"},
        "modalities": {"0": "US"},
        "dim": 3,
    }
    with open(task_dir / "dataset.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)
    logger.info(f"Wrote {task_dir / 'dataset.json'}")

    # ---- 7. Write dataset_statistics.csv ----
    csv_path = task_dir / "dataset_statistics.csv"
    fieldnames = [
        "volume_id", "source_folder", "image_path", "mask_path",
        "shape", "spacing", "num_lesions", "lesion_class", "split",
        "is_duplicate",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    logger.info(f"Wrote {csv_path}")

    # ---- Summary ----
    processed = [r for r in stats_rows if r["volume_id"] != "DUPLICATE"]
    logger.info(
        f"\nDone! {len(processed)} volumes processed, "
        f"{len(duplicates)} duplicates excluded."
    )
    logger.info(f"Task directory: {task_dir}")
    logger.info(
        f"Next steps:\n"
        f"  1. export det_data={det_data}\n"
        f"  2. nndet_prep 100 --full_check\n"
        f"  3. nndet_train 100 --fold 0\n"
    )


if __name__ == "__main__":
    main()

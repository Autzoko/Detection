"""
Extract 3D bounding boxes from nnDetection instance segmentation masks.

Outputs each lesion as (x1, y1, z1), (x2, y2, z2) in both voxel and
physical (mm) coordinates.

Usage:
    # Single case
    python scripts/extract_bboxes.py --mask /path/to/case_00001.nii.gz

    # Entire dataset
    python scripts/extract_bboxes.py \
        --label_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted/labelsTr" \
        --output bboxes.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def extract_bboxes_from_mask(mask_path: str) -> list:
    """
    Extract 3D bounding boxes from an instance segmentation mask.

    Returns list of dicts with voxel and physical coordinates.
    """
    mask_itk = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(mask_itk)  # shape: (Z, Y, X)
    spacing = mask_itk.GetSpacing()  # (sx, sy, sz)
    origin = mask_itk.GetOrigin()    # (ox, oy, oz)

    results = []
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue

        coords = np.argwhere(mask == instance_id)  # (Z, Y, X)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        # Voxel coordinates
        voxel_p1 = (int(x_min), int(y_min), int(z_min))
        voxel_p2 = (int(x_max), int(y_max), int(z_max))

        # Physical coordinates (mm)
        phys_p1 = (
            origin[0] + x_min * spacing[0],
            origin[1] + y_min * spacing[1],
            origin[2] + z_min * spacing[2],
        )
        phys_p2 = (
            origin[0] + x_max * spacing[0],
            origin[1] + y_max * spacing[1],
            origin[2] + z_max * spacing[2],
        )

        # Bounding box size
        size_voxel = (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
        size_mm = (
            size_voxel[0] * spacing[0],
            size_voxel[1] * spacing[1],
            size_voxel[2] * spacing[2],
        )

        results.append({
            "instance_id": int(instance_id),
            "voxel_p1": voxel_p1,
            "voxel_p2": voxel_p2,
            "phys_p1_mm": phys_p1,
            "phys_p2_mm": phys_p2,
            "size_voxel": size_voxel,
            "size_mm": size_mm,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D bounding boxes from instance masks"
    )
    parser.add_argument("--mask", type=str, help="Path to a single mask .nii.gz")
    parser.add_argument("--label_dir", type=str, help="Directory of mask .nii.gz files")
    parser.add_argument("--output", type=str, default="bboxes.csv", help="Output CSV path")
    args = parser.parse_args()

    if not args.mask and not args.label_dir:
        parser.error("Provide either --mask or --label_dir")

    # Collect mask files
    if args.mask:
        mask_files = [Path(args.mask)]
    else:
        label_dir = Path(args.label_dir)
        mask_files = sorted(label_dir.glob("*.nii.gz"))
        # Exclude any _0000.nii.gz (those are images)
        mask_files = [f for f in mask_files if "_0000" not in f.name]

    rows = []
    for mask_path in mask_files:
        case_id = mask_path.name.replace(".nii.gz", "")
        bboxes = extract_bboxes_from_mask(str(mask_path))

        if not bboxes:
            print(f"{case_id}: no lesions")
            continue

        for bb in bboxes:
            row = {
                "case_id": case_id,
                "instance_id": bb["instance_id"],
                "x1": bb["voxel_p1"][0],
                "y1": bb["voxel_p1"][1],
                "z1": bb["voxel_p1"][2],
                "x2": bb["voxel_p2"][0],
                "y2": bb["voxel_p2"][1],
                "z2": bb["voxel_p2"][2],
                "x1_mm": f"{bb['phys_p1_mm'][0]:.2f}",
                "y1_mm": f"{bb['phys_p1_mm'][1]:.2f}",
                "z1_mm": f"{bb['phys_p1_mm'][2]:.2f}",
                "x2_mm": f"{bb['phys_p2_mm'][0]:.2f}",
                "y2_mm": f"{bb['phys_p2_mm'][1]:.2f}",
                "z2_mm": f"{bb['phys_p2_mm'][2]:.2f}",
                "width": bb["size_voxel"][0],
                "height": bb["size_voxel"][1],
                "depth": bb["size_voxel"][2],
                "width_mm": f"{bb['size_mm'][0]:.2f}",
                "height_mm": f"{bb['size_mm'][1]:.2f}",
                "depth_mm": f"{bb['size_mm'][2]:.2f}",
            }
            rows.append(row)

            print(f"{case_id} lesion {bb['instance_id']}: "
                  f"voxel ({bb['voxel_p1']}) → ({bb['voxel_p2']}), "
                  f"size {bb['size_voxel']} voxels / "
                  f"({bb['size_mm'][0]:.1f}, {bb['size_mm'][1]:.1f}, {bb['size_mm'][2]:.1f}) mm")

    # Write CSV
    if rows:
        output_path = Path(args.output)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} bounding boxes to {output_path}")


if __name__ == "__main__":
    main()

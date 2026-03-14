"""
Visualize instance segmentation masks overlaid on the original image.

Shows three orthogonal views (axial, coronal, sagittal) through each lesion
center, with the ellipsoid mask contour and bounding box drawn on top.

Usage:
    # Visualize a single case
    python scripts/visualize_masks.py \
        --image "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted/imagesTr/case_00001_0000.nii.gz" \
        --mask "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/raw_splitted/labelsTr/case_00001.nii.gz" \
        --output_dir "./vis"

    # Visualize multiple cases from a dataset directory
    python scripts/visualize_masks.py \
        --data_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying" \
        --output_dir "./vis" \
        --num_cases 10

    # Visualize specific cases
    python scripts/visualize_masks.py \
        --data_dir "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying" \
        --output_dir "./vis" \
        --cases case_00001 case_00042 case_00100
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap


# Distinct colors for up to 10 lesion instances
INSTANCE_COLORS = [
    (1.0, 0.0, 0.0, 0.4),   # red
    (0.0, 1.0, 0.0, 0.4),   # green
    (0.0, 0.5, 1.0, 0.4),   # blue
    (1.0, 1.0, 0.0, 0.4),   # yellow
    (1.0, 0.0, 1.0, 0.4),   # magenta
    (0.0, 1.0, 1.0, 0.4),   # cyan
    (1.0, 0.5, 0.0, 0.4),   # orange
    (0.5, 0.0, 1.0, 0.4),   # purple
    (0.0, 1.0, 0.5, 0.4),   # spring green
    (1.0, 0.0, 0.5, 0.4),   # pink
]

CONTOUR_COLORS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.5, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 0.0, 1.0),
    (0.0, 1.0, 0.5),
    (1.0, 0.0, 0.5),
]


def load_volume(path: str):
    """Load a NIfTI volume, return numpy array and metadata."""
    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return data, {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "size": img.GetSize(),
    }


def get_lesion_info(mask: np.ndarray, spacing: tuple) -> list:
    """Get bounding box and center for each lesion instance."""
    lesions = []
    for lid in np.unique(mask):
        if lid == 0:
            continue
        coords = np.argwhere(mask == lid)  # (Z, Y, X)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        cz = (z_min + z_max) // 2
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        lesions.append({
            "id": int(lid),
            "center": (int(cz), int(cy), int(cx)),
            "bbox_min": (int(z_min), int(y_min), int(x_min)),
            "bbox_max": (int(z_max), int(y_max), int(x_max)),
            "voxel_count": int(coords.shape[0]),
            "size_mm": (
                (x_max - x_min + 1) * spacing[0],
                (y_max - y_min + 1) * spacing[1],
                (z_max - z_min + 1) * spacing[2],
            ),
        })
    return lesions


def visualize_case(image: np.ndarray, mask: np.ndarray, spacing: tuple,
                   case_id: str, output_dir: Path):
    """
    Create visualization for one case:
    - 3 orthogonal views through each lesion center
    - Overlay: mask contour + bounding box + transparent fill
    """
    lesions = get_lesion_info(mask, spacing)
    if not lesions:
        print(f"  {case_id}: no lesions found in mask")
        return

    n_lesions = len(lesions)

    # ── Figure 1: Overview — 3 views through the center of the volume ──
    vol_center = (image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{case_id} — Overview (volume center)\n"
                 f"Shape: {image.shape[::-1]} voxels, "
                 f"Spacing: ({spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f}) mm, "
                 f"Lesions: {n_lesions}",
                 fontsize=12, fontweight="bold")

    _draw_slice(axes[0], image, mask, "axial", vol_center[0], spacing, lesions)
    _draw_slice(axes[1], image, mask, "coronal", vol_center[1], spacing, lesions)
    _draw_slice(axes[2], image, mask, "sagittal", vol_center[2], spacing, lesions)

    plt.tight_layout()
    plt.savefig(str(output_dir / f"{case_id}_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Per-lesion views — 3 views through each lesion center ──
    for les in lesions:
        lid = les["id"]
        cz, cy, cx = les["center"]
        bmin = les["bbox_min"]
        bmax = les["bbox_max"]
        color_idx = (lid - 1) % len(INSTANCE_COLORS)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"{case_id} — Lesion {lid}\n"
            f"Center: ({cx}, {cy}, {cz}), "
            f"BBox: ({bmin[2]},{bmin[1]},{bmin[0]}) → ({bmax[2]},{bmax[1]},{bmax[0]}), "
            f"Size: {les['size_mm'][0]:.1f} x {les['size_mm'][1]:.1f} x {les['size_mm'][2]:.1f} mm, "
            f"Voxels: {les['voxel_count']}",
            fontsize=11, fontweight="bold"
        )

        # Row 1: Full slice with overlay
        _draw_slice(axes[0, 0], image, mask, "axial", cz, spacing, lesions,
                     highlight_id=lid)
        _draw_slice(axes[0, 1], image, mask, "coronal", cy, spacing, lesions,
                     highlight_id=lid)
        _draw_slice(axes[0, 2], image, mask, "sagittal", cx, spacing, lesions,
                     highlight_id=lid)

        # Row 2: Zoomed into lesion region (with padding)
        pad = 30
        _draw_slice(axes[1, 0], image, mask, "axial", cz, spacing, lesions,
                     highlight_id=lid, zoom=(bmin, bmax, pad))
        _draw_slice(axes[1, 1], image, mask, "coronal", cy, spacing, lesions,
                     highlight_id=lid, zoom=(bmin, bmax, pad))
        _draw_slice(axes[1, 2], image, mask, "sagittal", cx, spacing, lesions,
                     highlight_id=lid, zoom=(bmin, bmax, pad))

        plt.tight_layout()
        plt.savefig(str(output_dir / f"{case_id}_lesion{lid}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  {case_id}: {n_lesions} lesion(s) visualized")


def _draw_slice(ax, image, mask, view, slice_idx, spacing, lesions,
                highlight_id=None, zoom=None):
    """Draw a single 2D slice with mask overlay, contours, and bounding boxes."""

    if view == "axial":
        img_slice = image[slice_idx, :, :]       # Y x X
        mask_slice = mask[slice_idx, :, :]
        xlabel, ylabel = "X", "Y"
        title = f"Axial (Z={slice_idx})"
        aspect = spacing[1] / spacing[0]
    elif view == "coronal":
        img_slice = image[:, slice_idx, :]        # Z x X
        mask_slice = mask[:, slice_idx, :]
        xlabel, ylabel = "X", "Z"
        title = f"Coronal (Y={slice_idx})"
        aspect = spacing[2] / spacing[0]
    elif view == "sagittal":
        img_slice = image[:, :, slice_idx]        # Z x Y
        mask_slice = mask[:, :, slice_idx]
        xlabel, ylabel = "Y", "Z"
        title = f"Sagittal (X={slice_idx})"
        aspect = spacing[2] / spacing[1]

    # Display image
    ax.imshow(img_slice, cmap="gray", aspect=aspect, origin="upper")

    # Overlay each instance mask
    for les in lesions:
        lid = les["id"]
        color_idx = (lid - 1) % len(INSTANCE_COLORS)

        instance_slice = (mask_slice == lid).astype(float)
        if instance_slice.sum() == 0:
            continue

        # Transparent fill
        overlay = np.zeros((*instance_slice.shape, 4))
        overlay[instance_slice > 0] = INSTANCE_COLORS[color_idx]
        ax.imshow(overlay, aspect=aspect, origin="upper")

        # Contour
        ax.contour(instance_slice, levels=[0.5],
                   colors=[CONTOUR_COLORS[color_idx]], linewidths=1.5)

        # Bounding box on this slice
        bmin = les["bbox_min"]  # (z, y, x)
        bmax = les["bbox_max"]

        if view == "axial" and bmin[0] <= slice_idx <= bmax[0]:
            rect = mpatches.Rectangle(
                (bmin[2] - 0.5, bmin[1] - 0.5),
                bmax[2] - bmin[2] + 1, bmax[1] - bmin[1] + 1,
                linewidth=2 if lid == highlight_id else 1,
                edgecolor=CONTOUR_COLORS[color_idx],
                facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
        elif view == "coronal" and bmin[1] <= slice_idx <= bmax[1]:
            rect = mpatches.Rectangle(
                (bmin[2] - 0.5, bmin[0] - 0.5),
                bmax[2] - bmin[2] + 1, bmax[0] - bmin[0] + 1,
                linewidth=2 if lid == highlight_id else 1,
                edgecolor=CONTOUR_COLORS[color_idx],
                facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
        elif view == "sagittal" and bmin[2] <= slice_idx <= bmax[2]:
            rect = mpatches.Rectangle(
                (bmin[1] - 0.5, bmin[0] - 0.5),
                bmax[1] - bmin[1] + 1, bmax[0] - bmin[0] + 1,
                linewidth=2 if lid == highlight_id else 1,
                edgecolor=CONTOUR_COLORS[color_idx],
                facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)

    # Zoom if requested
    if zoom is not None:
        bmin, bmax, pad = zoom
        if view == "axial":
            ax.set_xlim(bmin[2] - pad, bmax[2] + pad)
            ax.set_ylim(bmax[1] + pad, bmin[1] - pad)
        elif view == "coronal":
            ax.set_xlim(bmin[2] - pad, bmax[2] + pad)
            ax.set_ylim(bmax[0] + pad, bmin[0] - pad)
        elif view == "sagittal":
            ax.set_xlim(bmin[1] - pad, bmax[1] + pad)
            ax.set_ylim(bmax[0] + pad, bmin[0] - pad)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize instance segmentation masks on breast ABUS volumes"
    )
    parser.add_argument("--image", type=str, help="Path to a single image .nii.gz")
    parser.add_argument("--mask", type=str, help="Path to a single mask .nii.gz")
    parser.add_argument("--data_dir", type=str,
                        help="nnDetection task dir (contains raw_splitted/)")
    parser.add_argument("--output_dir", type=str, default="./vis",
                        help="Output directory for visualizations")
    parser.add_argument("--num_cases", type=int, default=5,
                        help="Number of random cases to visualize (with --data_dir)")
    parser.add_argument("--cases", nargs="+", type=str,
                        help="Specific case IDs to visualize (e.g. case_00001)")
    parser.add_argument("--split", type=str, default="Tr",
                        choices=["Tr", "Ts"], help="Which split to use (Tr or Ts)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image and args.mask:
        # Single case mode
        print(f"Loading image: {args.image}")
        image, meta = load_volume(args.image)
        print(f"Loading mask: {args.mask}")
        mask, _ = load_volume(args.mask)
        case_id = Path(args.mask).name.replace(".nii.gz", "")
        visualize_case(image, mask, meta["spacing"], case_id, output_dir)

    elif args.data_dir:
        data_dir = Path(args.data_dir)
        img_dir = data_dir / "raw_splitted" / f"images{args.split}"
        lbl_dir = data_dir / "raw_splitted" / f"labels{args.split}"

        if not img_dir.exists():
            print(f"Image directory not found: {img_dir}")
            return

        # Find all cases
        all_images = sorted(img_dir.glob("*_0000.nii.gz"))
        all_case_ids = [f.name.replace("_0000.nii.gz", "") for f in all_images]

        # Select cases
        if args.cases:
            case_ids = [c for c in args.cases if c in all_case_ids]
            if not case_ids:
                print(f"None of the specified cases found. Available: {all_case_ids[:5]}...")
                return
        else:
            np.random.seed(42)
            indices = np.random.choice(len(all_case_ids),
                                       min(args.num_cases, len(all_case_ids)),
                                       replace=False)
            case_ids = [all_case_ids[i] for i in sorted(indices)]

        print(f"Visualizing {len(case_ids)} cases from {img_dir}")
        for case_id in case_ids:
            img_path = img_dir / f"{case_id}_0000.nii.gz"
            mask_path = lbl_dir / f"{case_id}.nii.gz"

            if not mask_path.exists():
                print(f"  {case_id}: mask not found, skipping")
                continue

            print(f"  Loading {case_id}...")
            image, meta = load_volume(str(img_path))
            mask, _ = load_volume(str(mask_path))
            visualize_case(image, mask, meta["spacing"], case_id, output_dir)

        print(f"\nAll visualizations saved to {output_dir}/")

    else:
        parser.error("Provide either --image/--mask or --data_dir")


if __name__ == "__main__":
    main()

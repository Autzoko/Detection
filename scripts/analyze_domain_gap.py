"""
Domain gap analysis between Duying (third-party) breast ABUS data and the
open-source TDSC-ABUS dataset (US43K/ABUS).

Compares:
  - Volume dimensions & physical size
  - Voxel spacing
  - Intensity distribution (mean, std, percentiles)
  - Lesion size & count
  - Image quality metrics (SNR, contrast)

Usage:
    python scripts/analyze_domain_gap.py \
        --duying_data "/Volumes/Autzoko/Dataset/third-party/data" \
        --abus_data "/Volumes/Autzoko/Dataset/US43K/ABUS/data" \
        --output_dir "./domain_analysis"

    # Or use already-prepared nnDetection data for Duying:
    python scripts/analyze_domain_gap.py \
        --duying_prepared "$det_data/Task100_BreastABUS" \
        --abus_data "/Volumes/Autzoko/Dataset/US43K/ABUS/data" \
        --output_dir "./domain_analysis"
"""

import argparse
import csv
import io
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import nrrd
    HAS_NRRD = True
except ImportError:
    HAS_NRRD = False

import SimpleITK as sitk
from loguru import logger

# ── Helpers ──────────────────────────────────────────────────────────────────


def read_nrrd_volume(path: str):
    """Read NRRD file, return numpy array and header dict."""
    if HAS_NRRD:
        data, header = nrrd.read(str(path))
        spacing = header.get("space directions", None)
        if spacing is not None:
            spacing = np.abs(np.diag(spacing[:3, :3])) if spacing.ndim == 2 else [1, 1, 1]
        else:
            spacing = [1.0, 1.0, 1.0]
        return data, {"spacing": spacing, "shape": data.shape}
    else:
        img = sitk.ReadImage(str(path))
        data = sitk.GetArrayFromImage(img)
        return data, {"spacing": list(img.GetSpacing()), "shape": list(img.GetSize())}


def read_nifti_volume(path: str):
    """Read NIfTI file, return numpy array and metadata."""
    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)
    return data, {
        "spacing": list(img.GetSpacing()),
        "shape": list(img.GetSize()),
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
    }


def compute_intensity_stats(data: np.ndarray) -> dict:
    """Compute intensity statistics on foreground voxels."""
    fg = data[data > 0].astype(np.float64)
    if len(fg) == 0:
        fg = data.astype(np.float64).ravel()
    return {
        "mean": float(np.mean(fg)),
        "std": float(np.std(fg)),
        "median": float(np.median(fg)),
        "p5": float(np.percentile(fg, 5)),
        "p25": float(np.percentile(fg, 25)),
        "p75": float(np.percentile(fg, 75)),
        "p95": float(np.percentile(fg, 95)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "dtype": str(data.dtype),
        "fg_fraction": float(len(fg)) / float(data.size),
    }


def compute_snr(data: np.ndarray) -> float:
    """Estimate SNR as mean(foreground) / std(background)."""
    fg_mask = data > np.percentile(data, 20)
    bg_mask = ~fg_mask
    fg_mean = np.mean(data[fg_mask].astype(np.float64)) if fg_mask.any() else 0
    bg_std = np.std(data[bg_mask].astype(np.float64)) if bg_mask.any() else 1
    return float(fg_mean / max(bg_std, 1e-8))


def compute_lesion_stats_from_mask(mask_data: np.ndarray) -> list:
    """Compute per-lesion size stats from an instance segmentation mask."""
    lesions = []
    for lid in np.unique(mask_data):
        if lid == 0:
            continue
        region = mask_data == lid
        coords = np.argwhere(region)
        bbox_size = coords.max(axis=0) - coords.min(axis=0) + 1
        lesions.append({
            "id": int(lid),
            "voxel_count": int(region.sum()),
            "bbox_size": bbox_size.tolist(),
        })
    return lesions


# ── ABUS (US43K) loader ─────────────────────────────────────────────────────


def load_abus_dataset(abus_root: Path, max_samples: int = 0) -> list:
    """Load ABUS dataset info from all splits."""
    records = []
    for split in ["Train", "Test", "Validation"]:
        split_dir = abus_root / split
        if not split_dir.exists():
            continue

        # Read labels
        labels_file = split_dir / "labels.csv"
        bbx_file = split_dir / "bbx_labels.csv"

        label_map = {}
        if labels_file.exists():
            with open(labels_file) as f:
                for row in csv.DictReader(f):
                    label_map[int(row["case_id"])] = row["label"]

        bbx_map = {}
        if bbx_file.exists():
            with open(bbx_file) as f:
                for row in csv.DictReader(f):
                    cid = int(row["id"])
                    bbx_map[cid] = {
                        "center": [float(row["c_x"]), float(row["c_y"]), float(row["c_z"])],
                        "half_len": [float(row["len_x"]), float(row["len_y"]), float(row["len_z"])],
                    }

        # Find data files
        data_dir = split_dir / "DATA"
        mask_dir = split_dir / "MASK"
        if not data_dir.exists():
            continue

        data_files = sorted(data_dir.glob("DATA_*.nrrd"))
        for df in data_files:
            case_id_str = df.stem.replace("DATA_", "")
            case_id = int(case_id_str)

            mask_file = mask_dir / f"MASK_{case_id_str}.nrrd"

            records.append({
                "dataset": "ABUS",
                "case_id": case_id,
                "split": split.lower(),
                "data_path": str(df),
                "mask_path": str(mask_file) if mask_file.exists() else None,
                "label": label_map.get(case_id, "unknown"),
                "bbx": bbx_map.get(case_id, None),
            })

            if max_samples and len(records) >= max_samples:
                return records

    return records


# ── Duying loader (raw) ─────────────────────────────────────────────────────

CLASS_FOLDERS = ["2类", "3类", "4类"]
TEST_FOLDER = "已标注及BI-rads分类20260123"
BIRADS_MAP = {"2类": "birads2", "3类": "birads3", "4类": "birads4"}


def _find_nii_pairs(folder: Path, lesion_class: str) -> list:
    """Find .nii + _nii_Label.tar pairs recursively."""
    pairs = []
    for nii_file in sorted(folder.rglob("*.nii")):
        if "_nii_Label" in nii_file.name:
            continue
        tar_name = nii_file.stem + "_nii_Label.tar"
        tar_file = nii_file.parent / tar_name
        if tar_file.exists():
            pairs.append({
                "image": nii_file,
                "label_tar": tar_file,
                "lesion_class": lesion_class,
            })
    return pairs


def parse_label_tar(tar_path: Path) -> dict:
    """Extract JSON annotation from label tar."""
    with tarfile.open(str(tar_path), "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                f = tf.extractfile(member)
                if f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    return {}


def count_bboxes_in_tar(tar_path: Path) -> int:
    """Count bounding box annotations in a label tar."""
    try:
        label_json = parse_label_tar(tar_path)
        bboxes = label_json.get("BoundingBoxLabelModel", [])
        # Group by lesion (unique spatial clusters)
        return max(1, len(set(bb.get("LabelId", 0) for bb in bboxes))) if bboxes else 0
    except Exception:
        return 0


def load_duying_raw(data_root: Path, max_samples: int = 0) -> list:
    """Load Duying dataset from raw folder structure."""
    records = []

    # Training class folders
    for cls_folder in CLASS_FOLDERS:
        cls_dir = data_root / cls_folder
        if not cls_dir.exists():
            continue
        pairs = _find_nii_pairs(cls_dir, BIRADS_MAP[cls_folder])
        for p in pairs:
            records.append({
                "dataset": "Duying",
                "data_path": str(p["image"]),
                "mask_path": str(p["label_tar"]),
                "lesion_class": p["lesion_class"],
                "split": "trainval",
                "is_tar": True,
            })
            if max_samples and len(records) >= max_samples:
                return records

    # Test set
    test_dir = data_root / TEST_FOLDER
    if test_dir.exists():
        pairs = _find_nii_pairs(test_dir, "test_mixed")
        for p in pairs:
            records.append({
                "dataset": "Duying",
                "data_path": str(p["image"]),
                "mask_path": str(p["label_tar"]),
                "lesion_class": p["lesion_class"],
                "split": "test",
                "is_tar": True,
            })
            if max_samples and len(records) >= max_samples:
                return records

    return records


# ── Duying loader (prepared nnDet format) ────────────────────────────────────


def load_duying_prepared(prepared_dir: Path, max_samples: int = 0) -> list:
    """Load Duying from already-prepared nnDetection Task directory."""
    records = []
    for subdir in ["imagesTr", "imagesTs"]:
        img_dir = prepared_dir / "raw_splitted" / subdir
        lbl_subdir = "labelsTr" if subdir == "imagesTr" else "labelsTs"
        lbl_dir = prepared_dir / "raw_splitted" / lbl_subdir
        split = "train" if "Tr" in subdir else "test"

        if not img_dir.exists():
            continue

        for img_file in sorted(img_dir.glob("*_0000.nii.gz")):
            case_id = img_file.name.replace("_0000.nii.gz", "")
            mask_file = lbl_dir / f"{case_id}.nii.gz"
            records.append({
                "dataset": "Duying",
                "case_id": case_id,
                "data_path": str(img_file),
                "mask_path": str(mask_file) if mask_file.exists() else None,
                "split": split,
                "is_tar": False,
            })
            if max_samples and len(records) >= max_samples:
                return records

    return records


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_volumes(records: list, dataset_name: str, max_analyze: int = 50) -> dict:
    """Analyze a set of volume records. Sample up to max_analyze for intensity."""
    logger.info(f"Analyzing {dataset_name}: {len(records)} total records, "
                f"sampling up to {max_analyze} for intensity analysis")

    stats = {
        "dataset": dataset_name,
        "total_cases": len(records),
        "shapes": [],
        "spacings": [],
        "physical_sizes": [],  # shape * spacing in mm
        "volumes_mm3": [],
        "intensity_stats": [],
        "snr_values": [],
        "lesion_counts": [],
        "lesion_voxel_sizes": [],
        "lesion_bbox_sizes": [],
        "dtypes": [],
    }

    sample_indices = np.random.choice(
        len(records), min(max_analyze, len(records)), replace=False
    )

    for idx, rec in enumerate(records):
        is_sampled = idx in sample_indices
        data_path = rec["data_path"]

        try:
            # Read volume
            if data_path.endswith(".nrrd"):
                data, meta = read_nrrd_volume(data_path)
            else:
                data, meta = read_nifti_volume(data_path)

            shape = meta["shape"]
            spacing = meta["spacing"]
            physical_size = [s * sp for s, sp in zip(shape, spacing)]
            volume_mm3 = float(np.prod(physical_size))

            stats["shapes"].append(shape)
            stats["spacings"].append(spacing)
            stats["physical_sizes"].append(physical_size)
            stats["volumes_mm3"].append(volume_mm3)

            # Intensity analysis (sampled)
            if is_sampled:
                int_stats = compute_intensity_stats(data)
                stats["intensity_stats"].append(int_stats)
                stats["dtypes"].append(int_stats["dtype"])
                stats["snr_values"].append(compute_snr(data))

            # Lesion analysis
            mask_path = rec.get("mask_path")
            if mask_path and os.path.exists(mask_path):
                if rec.get("is_tar", False):
                    n_lesions = count_bboxes_in_tar(Path(mask_path))
                    stats["lesion_counts"].append(n_lesions)
                elif mask_path.endswith(".nrrd"):
                    if is_sampled:
                        if HAS_NRRD:
                            mask_data, _ = nrrd.read(mask_path)
                        else:
                            mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                        lesions = compute_lesion_stats_from_mask(mask_data)
                        stats["lesion_counts"].append(len(lesions))
                        for les in lesions:
                            stats["lesion_voxel_sizes"].append(les["voxel_count"])
                            stats["lesion_bbox_sizes"].append(les["bbox_size"])
                    else:
                        # Just count from bbx info if available
                        if rec.get("bbx"):
                            stats["lesion_counts"].append(1)
                elif mask_path.endswith(".nii.gz"):
                    if is_sampled:
                        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                        lesions = compute_lesion_stats_from_mask(mask_data)
                        stats["lesion_counts"].append(len(lesions))
                        for les in lesions:
                            stats["lesion_voxel_sizes"].append(les["voxel_count"])
                            stats["lesion_bbox_sizes"].append(les["bbox_size"])

            # ABUS: use bbx_labels for lesion size
            if rec.get("bbx") and not stats["lesion_bbox_sizes"]:
                bbx = rec["bbx"]
                full_size = [2 * h for h in bbx["half_len"]]
                stats["lesion_bbox_sizes"].append(full_size)
                stats["lesion_counts"].append(1)

        except Exception as e:
            logger.warning(f"Failed to process {data_path}: {e}")
            continue

        if (idx + 1) % 20 == 0:
            logger.info(f"  Processed {idx + 1}/{len(records)}")

    return stats


# ── Reporting ────────────────────────────────────────────────────────────────


def summarize_list(values: list, name: str) -> dict:
    """Compute summary statistics for a list of numbers."""
    if not values:
        return {name: "N/A"}
    arr = np.array(values, dtype=float)
    return {
        f"{name}_mean": f"{np.mean(arr):.2f}",
        f"{name}_std": f"{np.std(arr):.2f}",
        f"{name}_median": f"{np.median(arr):.2f}",
        f"{name}_min": f"{np.min(arr):.2f}",
        f"{name}_max": f"{np.max(arr):.2f}",
    }


def generate_report(duying_stats: dict, abus_stats: dict, output_dir: Path):
    """Generate comparison report and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Text report ──
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DOMAIN GAP ANALYSIS: Duying vs TDSC-ABUS")
    report_lines.append("=" * 70)
    report_lines.append("")

    # 1. Dataset overview
    report_lines.append("## 1. Dataset Overview")
    report_lines.append(f"  Duying (third-party):  {duying_stats['total_cases']} volumes")
    report_lines.append(f"  TDSC-ABUS (US43K):     {abus_stats['total_cases']} volumes")
    report_lines.append("")

    # 2. Volume dimensions
    report_lines.append("## 2. Volume Dimensions (voxels)")
    for name, stats in [("Duying", duying_stats), ("ABUS", abus_stats)]:
        shapes = np.array(stats["shapes"])
        if len(shapes) > 0:
            report_lines.append(f"  {name}:")
            for ax, label in enumerate(["X (Width)", "Y (Height)", "Z (Depth)"]):
                vals = shapes[:, ax]
                report_lines.append(
                    f"    {label}: mean={np.mean(vals):.0f}, "
                    f"std={np.std(vals):.0f}, "
                    f"range=[{np.min(vals):.0f}, {np.max(vals):.0f}]"
                )
    report_lines.append("")

    # 3. Voxel spacing
    report_lines.append("## 3. Voxel Spacing (mm)")
    for name, stats in [("Duying", duying_stats), ("ABUS", abus_stats)]:
        spacings = np.array(stats["spacings"])
        if len(spacings) > 0:
            report_lines.append(f"  {name}:")
            for ax, label in enumerate(["X", "Y", "Z"]):
                vals = spacings[:, ax]
                report_lines.append(
                    f"    {label}: mean={np.mean(vals):.3f}, "
                    f"std={np.std(vals):.3f}, "
                    f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]"
                )
    report_lines.append("")

    # 4. Physical volume size
    report_lines.append("## 4. Physical Volume Size (mm³)")
    for name, stats in [("Duying", duying_stats), ("ABUS", abus_stats)]:
        phys = np.array(stats["physical_sizes"])
        if len(phys) > 0:
            report_lines.append(f"  {name}:")
            for ax, label in enumerate(["X", "Y", "Z"]):
                vals = phys[:, ax]
                report_lines.append(
                    f"    {label}: mean={np.mean(vals):.1f}mm, "
                    f"std={np.std(vals):.1f}mm, "
                    f"range=[{np.min(vals):.1f}, {np.max(vals):.1f}]mm"
                )
            vols = np.array(stats["volumes_mm3"])
            report_lines.append(
                f"    Total volume: mean={np.mean(vols)/1e6:.1f}cm³, "
                f"range=[{np.min(vols)/1e6:.1f}, {np.max(vols)/1e6:.1f}]cm³"
            )
    report_lines.append("")

    # 5. Intensity distribution
    report_lines.append("## 5. Intensity Distribution")
    for name, stats in [("Duying", duying_stats), ("ABUS", abus_stats)]:
        int_stats = stats["intensity_stats"]
        if int_stats:
            means = [s["mean"] for s in int_stats]
            stds = [s["std"] for s in int_stats]
            p5s = [s["p5"] for s in int_stats]
            p95s = [s["p95"] for s in int_stats]
            dtypes = list(set(s["dtype"] for s in int_stats))
            fg_fracs = [s["fg_fraction"] for s in int_stats]
            report_lines.append(f"  {name} (n={len(int_stats)} sampled):")
            report_lines.append(f"    Data type: {', '.join(dtypes)}")
            report_lines.append(
                f"    Mean intensity: {np.mean(means):.1f} ± {np.std(means):.1f}"
            )
            report_lines.append(
                f"    Std intensity:  {np.mean(stds):.1f} ± {np.std(stds):.1f}"
            )
            report_lines.append(
                f"    5th percentile: {np.mean(p5s):.1f} ± {np.std(p5s):.1f}"
            )
            report_lines.append(
                f"    95th percentile: {np.mean(p95s):.1f} ± {np.std(p95s):.1f}"
            )
            report_lines.append(
                f"    Foreground fraction: {np.mean(fg_fracs):.3f} ± {np.std(fg_fracs):.3f}"
            )
    report_lines.append("")

    # 6. SNR
    report_lines.append("## 6. Signal-to-Noise Ratio (estimated)")
    for name, stats in [("Duying", duying_stats), ("ABUS", abus_stats)]:
        snrs = stats["snr_values"]
        if snrs:
            report_lines.append(
                f"  {name}: mean={np.mean(snrs):.2f}, "
                f"std={np.std(snrs):.2f}, "
                f"range=[{np.min(snrs):.2f}, {np.max(snrs):.2f}]"
            )
    report_lines.append("")

    # 7. Lesion statistics
    report_lines.append("## 7. Lesion Statistics")
    for name, stats in [("Duying", duying_stats), ("ABUS", abus_stats)]:
        lc = stats["lesion_counts"]
        if lc:
            report_lines.append(f"  {name}:")
            report_lines.append(
                f"    Lesions per volume: mean={np.mean(lc):.2f}, "
                f"range=[{np.min(lc)}, {np.max(lc)}]"
            )
        lv = stats["lesion_voxel_sizes"]
        if lv:
            report_lines.append(
                f"    Lesion size (voxels): mean={np.mean(lv):.0f}, "
                f"median={np.median(lv):.0f}, "
                f"range=[{np.min(lv):.0f}, {np.max(lv):.0f}]"
            )
        lb = stats["lesion_bbox_sizes"]
        if lb:
            lb_arr = np.array(lb)
            report_lines.append(
                f"    Lesion bbox (voxels): "
                f"mean={np.mean(lb_arr, axis=0).tolist()}, "
                f"range X=[{np.min(lb_arr[:,0]):.0f},{np.max(lb_arr[:,0]):.0f}], "
                f"Y=[{np.min(lb_arr[:,1]):.0f},{np.max(lb_arr[:,1]):.0f}], "
                f"Z=[{np.min(lb_arr[:,2]):.0f},{np.max(lb_arr[:,2]):.0f}]"
            )
    report_lines.append("")

    # 8. Domain gap summary
    report_lines.append("## 8. Domain Gap Summary")
    report_lines.append("")

    # Spacing comparison
    d_spacings = np.array(duying_stats["spacings"]) if duying_stats["spacings"] else None
    a_spacings = np.array(abus_stats["spacings"]) if abus_stats["spacings"] else None
    if d_spacings is not None and a_spacings is not None:
        report_lines.append("  Spacing:")
        d_mean = np.mean(d_spacings, axis=0)
        a_mean = np.mean(a_spacings, axis=0)
        report_lines.append(f"    Duying mean: [{d_mean[0]:.3f}, {d_mean[1]:.3f}, {d_mean[2]:.3f}] mm")
        report_lines.append(f"    ABUS mean:   [{a_mean[0]:.3f}, {a_mean[1]:.3f}, {a_mean[2]:.3f}] mm")
        if not np.allclose(d_mean, a_mean, atol=0.1):
            report_lines.append("    ⚠ SIGNIFICANT spacing difference — resampling needed for cross-domain use")
        else:
            report_lines.append("    ✓ Similar spacing")

    # Intensity comparison
    d_int = duying_stats["intensity_stats"]
    a_int = abus_stats["intensity_stats"]
    if d_int and a_int:
        d_mean_int = np.mean([s["mean"] for s in d_int])
        a_mean_int = np.mean([s["mean"] for s in a_int])
        d_std_int = np.mean([s["std"] for s in d_int])
        a_std_int = np.mean([s["std"] for s in a_int])
        report_lines.append("  Intensity:")
        report_lines.append(f"    Duying: mean={d_mean_int:.1f}, std={d_std_int:.1f}")
        report_lines.append(f"    ABUS:   mean={a_mean_int:.1f}, std={a_std_int:.1f}")
        if abs(d_mean_int - a_mean_int) > 20 or abs(d_std_int - a_std_int) > 15:
            report_lines.append("    ⚠ SIGNIFICANT intensity distribution shift — normalization recommended")
        else:
            report_lines.append("    ✓ Similar intensity distribution")

    # Volume size comparison
    d_vols = duying_stats["volumes_mm3"]
    a_vols = abus_stats["volumes_mm3"]
    if d_vols and a_vols:
        report_lines.append("  Volume size:")
        report_lines.append(f"    Duying: mean={np.mean(d_vols)/1e6:.1f} cm³")
        report_lines.append(f"    ABUS:   mean={np.mean(a_vols)/1e6:.1f} cm³")
        ratio = np.mean(d_vols) / max(np.mean(a_vols), 1)
        if ratio > 2 or ratio < 0.5:
            report_lines.append(f"    ⚠ Volume size ratio: {ratio:.1f}x — significant FOV difference")
        else:
            report_lines.append(f"    ✓ Similar volume size (ratio: {ratio:.1f}x)")

    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = output_dir / "domain_gap_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_path}")

    # ── Plots ──
    generate_plots(duying_stats, abus_stats, output_dir)

    return report_text


def generate_plots(duying_stats: dict, abus_stats: dict, output_dir: Path):
    """Generate comparison plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Domain Gap Analysis: Duying vs TDSC-ABUS", fontsize=14, fontweight="bold")

    # 1. Volume dimensions comparison (box plots)
    ax = axes[0, 0]
    d_shapes = np.array(duying_stats["shapes"])
    a_shapes = np.array(abus_stats["shapes"])
    if len(d_shapes) > 0 and len(a_shapes) > 0:
        data_to_plot = []
        labels_to_plot = []
        for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
            data_to_plot.extend([d_shapes[:, ax_idx], a_shapes[:, ax_idx]])
            labels_to_plot.extend([f"Duying\n{ax_name}", f"ABUS\n{ax_name}"])
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        colors = ["#4C72B0", "#DD8452"] * 3
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_title("Volume Dimensions (voxels)")
    ax.set_ylabel("Voxels")

    # 2. Spacing comparison
    ax = axes[0, 1]
    d_spacings = np.array(duying_stats["spacings"])
    a_spacings = np.array(abus_stats["spacings"])
    if len(d_spacings) > 0 and len(a_spacings) > 0:
        x = np.arange(3)
        width = 0.35
        d_mean_sp = np.mean(d_spacings, axis=0)
        a_mean_sp = np.mean(a_spacings, axis=0)
        d_std_sp = np.std(d_spacings, axis=0)
        a_std_sp = np.std(a_spacings, axis=0)
        ax.bar(x - width / 2, d_mean_sp, width, yerr=d_std_sp,
               label="Duying", color="#4C72B0", alpha=0.7, capsize=3)
        ax.bar(x + width / 2, a_mean_sp, width, yerr=a_std_sp,
               label="ABUS", color="#DD8452", alpha=0.7, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["X", "Y", "Z"])
        ax.legend()
    ax.set_title("Voxel Spacing (mm)")
    ax.set_ylabel("mm")

    # 3. Intensity distribution
    ax = axes[0, 2]
    d_int = duying_stats["intensity_stats"]
    a_int = abus_stats["intensity_stats"]
    if d_int and a_int:
        d_means = [s["mean"] for s in d_int]
        a_means = [s["mean"] for s in a_int]
        ax.hist(d_means, bins=20, alpha=0.6, label="Duying", color="#4C72B0")
        ax.hist(a_means, bins=20, alpha=0.6, label="ABUS", color="#DD8452")
        ax.legend()
    ax.set_title("Mean Intensity Distribution")
    ax.set_xlabel("Mean Intensity")
    ax.set_ylabel("Count")

    # 4. Physical volume size
    ax = axes[1, 0]
    d_vols = np.array(duying_stats["volumes_mm3"]) / 1e6  # to cm³
    a_vols = np.array(abus_stats["volumes_mm3"]) / 1e6
    if len(d_vols) > 0 and len(a_vols) > 0:
        ax.hist(d_vols, bins=20, alpha=0.6, label="Duying", color="#4C72B0")
        ax.hist(a_vols, bins=20, alpha=0.6, label="ABUS", color="#DD8452")
        ax.legend()
    ax.set_title("Physical Volume Size")
    ax.set_xlabel("Volume (cm³)")
    ax.set_ylabel("Count")

    # 5. SNR comparison
    ax = axes[1, 1]
    d_snr = duying_stats["snr_values"]
    a_snr = abus_stats["snr_values"]
    if d_snr and a_snr:
        ax.hist(d_snr, bins=15, alpha=0.6, label="Duying", color="#4C72B0")
        ax.hist(a_snr, bins=15, alpha=0.6, label="ABUS", color="#DD8452")
        ax.legend()
    ax.set_title("Estimated SNR")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Count")

    # 6. Lesion size
    ax = axes[1, 2]
    d_lesion_v = duying_stats["lesion_voxel_sizes"]
    a_lesion_v = abus_stats["lesion_voxel_sizes"]
    if d_lesion_v or a_lesion_v:
        plot_data = []
        plot_labels = []
        if d_lesion_v:
            plot_data.append(np.log10(np.array(d_lesion_v) + 1))
            plot_labels.append("Duying")
        if a_lesion_v:
            plot_data.append(np.log10(np.array(a_lesion_v) + 1))
            plot_labels.append("ABUS")
        colors = ["#4C72B0", "#DD8452"][:len(plot_data)]
        for d, l, c in zip(plot_data, plot_labels, colors):
            ax.hist(d, bins=20, alpha=0.6, label=l, color=c)
        ax.legend()
    ax.set_title("Lesion Size (log₁₀ voxels)")
    ax.set_xlabel("log₁₀(voxel count)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plot_path = output_dir / "domain_gap_comparison.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plots saved to {plot_path}")

    # ── Per-axis physical size comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Physical Size per Axis (mm)", fontsize=12)
    d_phys = np.array(duying_stats["physical_sizes"])
    a_phys = np.array(abus_stats["physical_sizes"])
    for ax_idx, (ax, label) in enumerate(zip(axes, ["X", "Y", "Z"])):
        if len(d_phys) > 0:
            ax.hist(d_phys[:, ax_idx], bins=20, alpha=0.6, label="Duying", color="#4C72B0")
        if len(a_phys) > 0:
            ax.hist(a_phys[:, ax_idx], bins=20, alpha=0.6, label="ABUS", color="#DD8452")
        ax.set_title(f"{label} axis")
        ax.set_xlabel("mm")
        ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "physical_size_per_axis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Lesion bbox per-axis comparison ──
    d_lb = duying_stats["lesion_bbox_sizes"]
    a_lb = abus_stats["lesion_bbox_sizes"]
    if d_lb or a_lb:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Lesion Bounding Box Size per Axis (voxels)", fontsize=12)
        for ax_idx, (ax, label) in enumerate(zip(axes, ["X", "Y", "Z"])):
            if d_lb:
                d_arr = np.array(d_lb)
                ax.hist(d_arr[:, ax_idx], bins=20, alpha=0.6, label="Duying", color="#4C72B0")
            if a_lb:
                a_arr = np.array(a_lb)
                ax.hist(a_arr[:, ax_idx], bins=20, alpha=0.6, label="ABUS", color="#DD8452")
            ax.set_title(f"{label} axis")
            ax.set_xlabel("voxels")
            ax.legend()
        plt.tight_layout()
        plt.savefig(str(output_dir / "lesion_bbox_per_axis.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Domain gap analysis: Duying vs TDSC-ABUS"
    )
    parser.add_argument(
        "--duying_data", type=str, default=None,
        help="Path to raw Duying data root (third-party/data)"
    )
    parser.add_argument(
        "--duying_prepared", type=str, default=None,
        help="Path to prepared nnDetection Task directory (alternative to --duying_data)"
    )
    parser.add_argument(
        "--abus_data", type=str, required=True,
        help="Path to TDSC-ABUS data root (US43K/ABUS/data)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./domain_analysis",
        help="Output directory for report and plots"
    )
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Max volumes to sample for intensity analysis per dataset"
    )
    args = parser.parse_args()

    if not args.duying_data and not args.duying_prepared:
        parser.error("Provide either --duying_data or --duying_prepared")

    output_dir = Path(args.output_dir)
    abus_root = Path(args.abus_data)

    # Load ABUS
    logger.info("Loading ABUS dataset records...")
    abus_records = load_abus_dataset(abus_root)
    logger.info(f"ABUS: {len(abus_records)} records")

    # Load Duying
    if args.duying_prepared:
        logger.info("Loading Duying from prepared nnDetection format...")
        duying_records = load_duying_prepared(Path(args.duying_prepared))
    else:
        logger.info("Loading Duying from raw data...")
        duying_records = load_duying_raw(Path(args.duying_data))
    logger.info(f"Duying: {len(duying_records)} records")

    # Analyze
    np.random.seed(42)
    duying_stats = analyze_volumes(duying_records, "Duying", max_analyze=args.max_samples)
    abus_stats = analyze_volumes(abus_records, "ABUS", max_analyze=args.max_samples)

    # Report
    generate_report(duying_stats, abus_stats, output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()

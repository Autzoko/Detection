"""
Compare baseline (from Prediction_vs_GT_Comparison.xlsx) vs nnDetection post-processed results.

Produces:
  1. Per-case bar chart comparison (precision, recall, F1)
  2. Aggregate metrics comparison
  3. FP/FN comparison
  4. Scatter: baseline vs postproc per-case metrics
  5. Per-case detailed table
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import SimpleITK as sitk
import yaml


# ============================================================
# Data loading
# ============================================================
def extract_gt_boxes(label_path, json_path=None):
    label_sitk = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(label_sitk)
    gt_list = []
    for label_val in sorted(v for v in np.unique(arr) if v > 0):
        coords = np.argwhere(arr == label_val)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        box = [mins[0], mins[1], maxs[0], maxs[1], mins[2], maxs[2]]
        gt_list.append({"box": box})
    return gt_list


def iou_3d(box1, box2):
    z1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1]); x1 = max(box1[4], box2[4])
    z2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3]); x2 = min(box1[5], box2[5])
    inter = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    vol1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    vol2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def compute_case_metrics(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.1):
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_pred == 0:
        return {"TP": 0, "FP": 0, "FN": n_gt, "n_pred": 0, "n_gt": n_gt,
                "precision": 0, "recall": 0, "f1": 0}
    sorted_idx = np.argsort(-pred_scores)
    gt_matched = [False] * n_gt
    tp = 0
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
            tp += 1
            gt_matched[best_gi] = True
    fp = n_pred - tp
    fn = n_gt - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / n_gt if n_gt > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"TP": tp, "FP": fp, "FN": fn, "n_pred": n_pred, "n_gt": n_gt,
            "precision": p, "recall": r, "f1": f1}


def build_case_mapping(stats_csv):
    """Map case_id -> original filename stem."""
    stats = pd.read_csv(stats_csv)
    test_cases = stats[stats['split'] == 'test'].drop_duplicates('volume_id')
    mapping = {}
    for _, row in test_cases.iterrows():
        stem = os.path.basename(row['image_path']).replace('.nii', '')
        mapping[row['volume_id']] = stem
    return mapping


def main():
    # Paths
    config_path = Path(__file__).parent / "patch_classifier" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    labels_dir = cfg["paths"]["test_labels_dir"]
    pp_cfg = cfg["postprocessing"]
    pp_base = pp_cfg["output_dir"]

    # Find post-processed predictions
    pp_subdirs = [d for d in Path(pp_base).iterdir() if d.is_dir() and d.name.startswith("cluster")]
    pp_pred_dir = sorted(pp_subdirs)[-1]
    print(f"Post-processed predictions: {pp_pred_dir}")

    save_dir = os.path.join(pp_base, "baseline_comparison")
    os.makedirs(save_dir, exist_ok=True)

    # Load baseline
    baseline_xlsx = "/Users/langtian/Desktop/Prediction_vs_GT_Comparison.xlsx"
    bl_per_file = pd.read_excel(baseline_xlsx, sheet_name="Per-File Summary")
    bl_detail = pd.read_excel(baseline_xlsx, sheet_name="Detailed Comparison")

    # Build mapping
    stats_csv = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/dataset_statistics.csv"
    case_mapping = build_case_mapping(stats_csv)
    # Reverse: stem -> case_id
    stem_to_case = {v: k for k, v in case_mapping.items()}

    # ---- Compute per-case metrics for post-processed ----
    pp_cases = []
    for _, bl_row in bl_per_file.iterrows():
        bl_fname = bl_row["Filename"]
        stem = bl_fname.replace(".ai", "")
        case_id = stem_to_case.get(stem)
        if case_id is None:
            print(f"Warning: no case_id for {bl_fname}")
            continue

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

        # Load GT
        gt_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        json_path = os.path.join(labels_dir, f"{case_id}.json")
        gt_list = extract_gt_boxes(gt_path, json_path)
        gt_box_list = [g["box"] for g in gt_list]

        pp_metrics = compute_case_metrics(pp_boxes, pp_scores, gt_box_list)

        pp_cases.append({
            "case_id": case_id,
            "filename": bl_fname,
            "stem": stem,
            "n_gt": bl_row["# GT Lesions"],
            # Baseline
            "bl_n_pred": bl_row["# Predictions"],
            "bl_tp": bl_row["# TP (Matched)"],
            "bl_fp": bl_row["# FP"],
            "bl_fn": bl_row["# FN"],
            "bl_precision": bl_row["Det. Precision"],
            "bl_recall": bl_row["Det. Recall"],
            "bl_f1": bl_row["Det. F1"] if not pd.isna(bl_row["Det. F1"]) else 0,
            # Post-processed
            "pp_n_pred": pp_metrics["n_pred"],
            "pp_tp": pp_metrics["TP"],
            "pp_fp": pp_metrics["FP"],
            "pp_fn": pp_metrics["FN"],
            "pp_precision": pp_metrics["precision"],
            "pp_recall": pp_metrics["recall"],
            "pp_f1": pp_metrics["f1"],
        })

    df = pd.DataFrame(pp_cases)

    # ---- Aggregate metrics ----
    bl_total_tp = df["bl_tp"].sum()
    bl_total_fp = df["bl_fp"].sum()
    bl_total_fn = df["bl_fn"].sum()
    bl_total_gt = df["n_gt"].sum()
    bl_prec = bl_total_tp / (bl_total_tp + bl_total_fp) if (bl_total_tp + bl_total_fp) > 0 else 0
    bl_rec = bl_total_tp / bl_total_gt if bl_total_gt > 0 else 0
    bl_f1 = 2 * bl_prec * bl_rec / (bl_prec + bl_rec) if (bl_prec + bl_rec) > 0 else 0

    pp_total_tp = df["pp_tp"].sum()
    pp_total_fp = df["pp_fp"].sum()
    pp_total_fn = df["pp_fn"].sum()
    pp_prec = pp_total_tp / (pp_total_tp + pp_total_fp) if (pp_total_tp + pp_total_fp) > 0 else 0
    pp_rec = pp_total_tp / bl_total_gt if bl_total_gt > 0 else 0
    pp_f1 = 2 * pp_prec * pp_rec / (pp_prec + pp_rec) if (pp_prec + pp_rec) > 0 else 0

    n_cases = len(df)

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE COMPARISON ({n_cases} cases, {bl_total_gt} GT lesions)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<20} {'Baseline':>12} {'PostProc':>12} {'Delta':>12}")
    print("-" * 56)
    for label, bv, pv in [
        ("TP", bl_total_tp, pp_total_tp),
        ("FP", bl_total_fp, pp_total_fp),
        ("FN", bl_total_fn, pp_total_fn),
        ("Predictions", df["bl_n_pred"].sum(), df["pp_n_pred"].sum()),
        ("Precision", bl_prec, pp_prec),
        ("Recall", bl_rec, pp_rec),
        ("F1", bl_f1, pp_f1),
        ("FP/case", bl_total_fp / n_cases, pp_total_fp / n_cases),
    ]:
        if isinstance(bv, float):
            print(f"{label:<20} {bv:>12.4f} {pv:>12.4f} {pv - bv:>+12.4f}")
        else:
            print(f"{label:<20} {bv:>12} {pv:>12} {pv - bv:>+12}")
    print(f"{'=' * 60}")

    # ============================================================
    # Plot 1: Aggregate metrics comparison (grouped bar)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_labels = ["Precision", "Recall", "F1"]
    bl_vals = [bl_prec, bl_rec, bl_f1]
    pp_vals = [pp_prec, pp_rec, pp_f1]

    x = np.arange(len(metrics_labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, bl_vals, w, label="Baseline", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + w / 2, pp_vals, w, label="nnDetection + PostProc", color="#C44E52", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline vs nnDetection Post-Processed: Detection Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "aggregate_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregate_metrics.png")

    # ============================================================
    # Plot 2: TP / FP / FN comparison (stacked bar)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["Baseline", "nnDet PostProc"]
    tp_vals = [bl_total_tp, pp_total_tp]
    fp_vals = [bl_total_fp, pp_total_fp]
    fn_vals = [bl_total_fn, pp_total_fn]

    x = np.arange(2)
    w = 0.5
    ax.bar(x, tp_vals, w, label="TP", color="#55A868")
    ax.bar(x, fp_vals, w, bottom=tp_vals, label="FP", color="#C44E52", alpha=0.8)
    ax.bar(x, fn_vals, w, bottom=[t + f for t, f in zip(tp_vals, fp_vals)],
           label="FN", color="#DD8452", alpha=0.8)

    # Annotate
    for i in range(2):
        ax.text(x[i], tp_vals[i] / 2, f"TP={tp_vals[i]}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x[i], tp_vals[i] + fp_vals[i] / 2, f"FP={fp_vals[i]}", ha="center",
                va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(x[i], tp_vals[i] + fp_vals[i] + fn_vals[i] / 2, f"FN={fn_vals[i]}",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("TP / FP / FN Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "tp_fp_fn_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tp_fp_fn_comparison.png")

    # ============================================================
    # Plot 3: Per-case F1 comparison (paired bar chart, sorted by baseline F1)
    # ============================================================
    df_sorted = df.sort_values("bl_f1", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(n_cases)
    h = 0.4
    ax.barh(y - h / 2, df_sorted["bl_f1"], h, label="Baseline", color="#4C72B0", alpha=0.8)
    ax.barh(y + h / 2, df_sorted["pp_f1"], h, label="nnDet PostProc", color="#C44E52", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["stem"], fontsize=6)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Case F1: Baseline vs nnDetection Post-Processed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_case_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per_case_f1.png")

    # ============================================================
    # Plot 4: Per-case scatter (baseline F1 vs postproc F1)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["bl_f1"], df["pp_f1"], s=60, alpha=0.7, edgecolor="black", linewidth=0.5,
               c=df["n_gt"], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Equal performance")
    ax.set_xlabel("Baseline F1", fontsize=12)
    ax.set_ylabel("nnDet PostProc F1", fontsize=12)
    ax.set_title("Per-Case F1 Scatter", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("# GT lesions", fontsize=10)
    # Count above/below diagonal
    above = (df["pp_f1"] > df["bl_f1"]).sum()
    below = (df["pp_f1"] < df["bl_f1"]).sum()
    equal = (df["pp_f1"] == df["bl_f1"]).sum()
    ax.text(0.05, 0.95, f"PostProc better: {above}\nBaseline better: {below}\nEqual: {equal}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "f1_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved f1_scatter.png")

    # ============================================================
    # Plot 5: Per-case FP comparison
    # ============================================================
    df_sorted_fp = df.sort_values("bl_fp", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(n_cases)
    h = 0.4
    ax.barh(y - h / 2, df_sorted_fp["bl_fp"], h, label="Baseline FP", color="#C44E52", alpha=0.8)
    ax.barh(y + h / 2, df_sorted_fp["pp_fp"], h, label="PostProc FP", color="#4C72B0", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted_fp["stem"], fontsize=6)
    ax.set_xlabel("False Positives", fontsize=12)
    ax.set_title("Per-Case FP Count: Baseline vs nnDetection Post-Processed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_case_fp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per_case_fp.png")

    # ============================================================
    # Plot 6: Per-case recall comparison
    # ============================================================
    df_sorted_r = df.sort_values("bl_recall", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(n_cases)
    h = 0.4
    ax.barh(y - h / 2, df_sorted_r["bl_recall"], h, label="Baseline Recall", color="#55A868", alpha=0.8)
    ax.barh(y + h / 2, df_sorted_r["pp_recall"], h, label="PostProc Recall", color="#DD8452", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted_r["stem"], fontsize=6)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_title("Per-Case Recall: Baseline vs nnDetection Post-Processed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_case_recall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per_case_recall.png")

    # ============================================================
    # Plot 7: Summary dashboard (2x2)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline vs nnDetection Post-Processed: Summary Dashboard",
                 fontsize=15, fontweight="bold")

    # (0,0) Aggregate bars
    ax = axes[0, 0]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, bl_vals, w, label="Baseline", color="#4C72B0")
    ax.bar(x + w / 2, pp_vals, w, label="PostProc", color="#C44E52")
    for i in range(3):
        ax.text(x[i] - w / 2, bl_vals[i] + 0.02, f"{bl_vals[i]:.3f}", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, pp_vals[i] + 0.02, f"{pp_vals[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Aggregate Metrics")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (0,1) TP/FP/FN
    ax = axes[0, 1]
    x = np.arange(2)
    w = 0.5
    ax.bar(x, tp_vals, w, label="TP", color="#55A868")
    ax.bar(x, fp_vals, w, bottom=tp_vals, label="FP", color="#C44E52", alpha=0.8)
    ax.bar(x, fn_vals, w, bottom=[t + f for t, f in zip(tp_vals, fp_vals)],
           label="FN", color="#DD8452", alpha=0.8)
    for i in range(2):
        ax.text(x[i], tp_vals[i] / 2, f"TP={tp_vals[i]}", ha="center", va="center", fontsize=10, color="white")
        if fp_vals[i] > 5:
            ax.text(x[i], tp_vals[i] + fp_vals[i] / 2, f"FP={fp_vals[i]}", ha="center", va="center", fontsize=10, color="white")
        if fn_vals[i] > 3:
            ax.text(x[i], tp_vals[i] + fp_vals[i] + fn_vals[i] / 2, f"FN={fn_vals[i]}", ha="center", va="center", fontsize=10, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "PostProc"], fontsize=11)
    ax.set_title("Detection Counts")
    ax.legend(fontsize=9)

    # (1,0) F1 scatter
    ax = axes[1, 0]
    sc = ax.scatter(df["bl_f1"], df["pp_f1"], s=40, alpha=0.7, edgecolor="black",
                    linewidth=0.5, c=df["n_gt"], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Baseline F1")
    ax.set_ylabel("PostProc F1")
    ax.set_title("Per-Case F1 Scatter")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f"Better: {above} | Worse: {below} | Equal: {equal}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.colorbar(sc, ax=ax, label="# GT")

    # (1,1) FP distribution
    ax = axes[1, 1]
    bins = np.arange(0, max(df["bl_fp"].max(), df["pp_fp"].max()) + 2) - 0.5
    ax.hist(df["bl_fp"], bins=bins, alpha=0.6, label="Baseline", color="#C44E52", edgecolor="white")
    ax.hist(df["pp_fp"], bins=bins, alpha=0.6, label="PostProc", color="#4C72B0", edgecolor="white")
    ax.set_xlabel("FP per case")
    ax.set_ylabel("Number of cases")
    ax.set_title("FP Distribution")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary_dashboard.png")

    # ============================================================
    # Save per-case comparison CSV
    # ============================================================
    csv_path = os.path.join(save_dir, "per_case_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ============================================================
    # Print per-case table
    # ============================================================
    print(f"\n{'=' * 100}")
    print(f"  PER-CASE COMPARISON TABLE")
    print(f"{'=' * 100}")
    header = (f"{'Case':<15} {'GT':>3} {'BL_P':>6} {'BL_R':>6} {'BL_F1':>6} {'BL_TP':>5} {'BL_FP':>5} {'BL_FN':>5} "
              f"{'PP_P':>6} {'PP_R':>6} {'PP_F1':>6} {'PP_TP':>5} {'PP_FP':>5} {'PP_FN':>5} {'Winner':>10}")
    print(header)
    print("-" * 100)
    for _, r in df.iterrows():
        winner = "PostProc" if r["pp_f1"] > r["bl_f1"] else ("Baseline" if r["pp_f1"] < r["bl_f1"] else "Tie")
        print(f"{r['stem']:<15} {r['n_gt']:>3} "
              f"{r['bl_precision']:>6.3f} {r['bl_recall']:>6.3f} {r['bl_f1']:>6.3f} "
              f"{r['bl_tp']:>5} {r['bl_fp']:>5} {r['bl_fn']:>5} "
              f"{r['pp_precision']:>6.3f} {r['pp_recall']:>6.3f} {r['pp_f1']:>6.3f} "
              f"{r['pp_tp']:>5} {r['pp_fp']:>5} {r['pp_fn']:>5} "
              f"{winner:>10}")
    print(f"{'=' * 100}")

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()

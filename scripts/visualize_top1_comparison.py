"""
Compare baseline vs nnDetection keeping only the top-1 highest confidence
prediction per case. Visualize the comparison.
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import SimpleITK as sitk
import yaml


def extract_gt_boxes(label_path, json_path=None):
    label_sitk = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(label_sitk)
    gt_list = []
    for label_val in sorted(v for v in np.unique(arr) if v > 0):
        coords = np.argwhere(arr == label_val)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        box = [mins[0], mins[1], maxs[0], maxs[1], mins[2], maxs[2]]
        gt_list.append(box)
    return gt_list


def iou_3d(box1, box2):
    z1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1]); x1 = max(box1[4], box2[4])
    z2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3]); x2 = min(box1[5], box2[5])
    inter = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    vol1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) * (box1[5] - box1[4])
    vol2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) * (box2[5] - box2[4])
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def match_top_n(pred_boxes, pred_scores, gt_boxes, n, iou_thresh=0.1):
    """Keep top-n predictions by score, return TP/FP/FN."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    top_idx = np.argsort(-pred_scores)[:n]
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    for pi in top_idx:
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
    fp = len(top_idx) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def build_case_mapping(stats_csv):
    stats = pd.read_csv(stats_csv)
    test_cases = stats[stats['split'] == 'test'].drop_duplicates('volume_id')
    mapping = {}
    for _, row in test_cases.iterrows():
        stem = os.path.basename(row['image_path']).replace('.nii', '')
        mapping[row['volume_id']] = stem
    return mapping


def main():
    config_path = Path(__file__).parent / "patch_classifier" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pred_dir = cfg["paths"]["test_predictions_dir"]
    labels_dir = cfg["paths"]["test_labels_dir"]
    pp_cfg = cfg["postprocessing"]
    pp_base = pp_cfg["output_dir"]

    save_dir = os.path.join(pp_base, "top1_comparison")
    os.makedirs(save_dir, exist_ok=True)

    # Load baseline
    baseline_xlsx = "/Users/langtian/Desktop/Prediction_vs_GT_Comparison.xlsx"
    bl_per_file = pd.read_excel(baseline_xlsx, sheet_name="Per-File Summary")

    # Build mapping
    stats_csv = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/dataset_statistics.csv"
    case_mapping = build_case_mapping(stats_csv)
    stem_to_case = {v: k for k, v in case_mapping.items()}

    # Sweep top-N from 1 to 10
    top_n_range = list(range(1, 11))

    # Collect per-case results for each top-N
    rows = []

    for _, bl_row in bl_per_file.iterrows():
        bl_fname = bl_row["Filename"]
        stem = bl_fname.replace(".ai", "")
        case_id = stem_to_case.get(stem)
        if case_id is None:
            continue

        # Load raw nnDetection predictions (all, no postprocessing)
        pkl_path = os.path.join(pred_dir, f"{case_id}_boxes.pkl")
        with open(pkl_path, "rb") as f:
            pred = pickle.load(f)
        pred_boxes = pred["pred_boxes"]
        pred_scores = pred["pred_scores"]

        # Load GT
        gt_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        json_path = os.path.join(labels_dir, f"{case_id}.json")
        gt_boxes = extract_gt_boxes(gt_path, json_path)
        n_gt = len(gt_boxes)

        for n in top_n_range:
            tp, fp, fn = match_top_n(pred_boxes, pred_scores, gt_boxes, n)
            rows.append({
                "case_id": case_id, "stem": stem, "n_gt": n_gt,
                "top_n": n, "tp": tp, "fp": fp, "fn": fn,
                # baseline (same for all n)
                "bl_tp": bl_row["# TP (Matched)"],
                "bl_fp": bl_row["# FP"],
                "bl_fn": bl_row["# FN"],
                "bl_n_pred": bl_row["# Predictions"],
            })

    df = pd.DataFrame(rows)
    n_cases = df["case_id"].nunique()
    total_gt = df[df["top_n"] == 1]["n_gt"].sum()

    # Baseline aggregate (constant)
    bl_tp_tot = df[df["top_n"] == 1]["bl_tp"].sum()
    bl_fp_tot = df[df["top_n"] == 1]["bl_fp"].sum()
    bl_fn_tot = df[df["top_n"] == 1]["bl_fn"].sum()
    bl_prec = bl_tp_tot / (bl_tp_tot + bl_fp_tot) if (bl_tp_tot + bl_fp_tot) > 0 else 0
    bl_rec = bl_tp_tot / total_gt if total_gt > 0 else 0
    bl_f1 = 2 * bl_prec * bl_rec / (bl_prec + bl_rec) if (bl_prec + bl_rec) > 0 else 0

    # Aggregate per top-N
    sweep = []
    for n in top_n_range:
        sub = df[df["top_n"] == n]
        tp = sub["tp"].sum()
        fp = sub["fp"].sum()
        fn = sub["fn"].sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / total_gt if total_gt > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        sweep.append({"top_n": n, "TP": tp, "FP": fp, "FN": fn,
                       "precision": p, "recall": r, "f1": f1,
                       "fp_per_case": fp / n_cases, "n_pred": n * n_cases})

    # Print table
    print(f"\n{'=' * 80}")
    print(f"  TOP-N SWEEP vs BASELINE  ({n_cases} cases, {total_gt} GT)")
    print(f"  Baseline: P={bl_prec:.3f}, R={bl_rec:.3f}, F1={bl_f1:.3f}, "
          f"TP={bl_tp_tot}, FP={bl_fp_tot}, FN={bl_fn_tot}, FP/case={bl_fp_tot/n_cases:.2f}")
    print(f"{'=' * 80}")
    print(f"{'Top-N':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'FP/case':>10}")
    print("-" * 60)
    for s in sweep:
        marker = ""
        if s["f1"] >= bl_f1:
            marker = " *** beats baseline"
        print(f"{s['top_n']:>6} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} "
              f"{s['TP']:>5} {s['FP']:>5} {s['FN']:>5} {s['fp_per_case']:>10.2f}{marker}")
    print(f"{'=' * 80}")

    # ---- Per-case table for top-1 ----
    top1 = df[df["top_n"] == 1].copy()
    top1["pp_prec"] = top1.apply(lambda r: r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) > 0 else 0, axis=1)
    top1["pp_rec"] = top1.apply(lambda r: r["tp"] / r["n_gt"] if r["n_gt"] > 0 else 0, axis=1)
    top1["pp_f1"] = top1.apply(lambda r: 2*r["pp_prec"]*r["pp_rec"]/(r["pp_prec"]+r["pp_rec"]) if (r["pp_prec"]+r["pp_rec"])>0 else 0, axis=1)
    top1["bl_prec_c"] = top1.apply(lambda r: r["bl_tp"]/(r["bl_tp"]+r["bl_fp"]) if (r["bl_tp"]+r["bl_fp"])>0 else 0, axis=1)
    top1["bl_rec_c"] = top1.apply(lambda r: r["bl_tp"]/r["n_gt"] if r["n_gt"]>0 else 0, axis=1)
    top1["bl_f1_c"] = top1.apply(lambda r: 2*r["bl_prec_c"]*r["bl_rec_c"]/(r["bl_prec_c"]+r["bl_rec_c"]) if (r["bl_prec_c"]+r["bl_rec_c"])>0 else 0, axis=1)

    # ============================================================
    # Plot 1: Top-N sweep line chart
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("nnDetection Top-N Predictions vs Baseline", fontsize=14, fontweight="bold")

    ns = [s["top_n"] for s in sweep]
    ax1.plot(ns, [s["precision"] for s in sweep], "o-", label="nnDet Precision", color="#4C72B0")
    ax1.plot(ns, [s["recall"] for s in sweep], "s-", label="nnDet Recall", color="#55A868")
    ax1.plot(ns, [s["f1"] for s in sweep], "^-", label="nnDet F1", color="#C44E52")
    ax1.axhline(bl_prec, color="#4C72B0", linestyle="--", alpha=0.5, label=f"BL Prec ({bl_prec:.3f})")
    ax1.axhline(bl_rec, color="#55A868", linestyle="--", alpha=0.5, label=f"BL Recall ({bl_rec:.3f})")
    ax1.axhline(bl_f1, color="#C44E52", linestyle="--", alpha=0.5, label=f"BL F1 ({bl_f1:.3f})")
    ax1.set_xlabel("Top-N predictions kept per case")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1")
    ax1.set_xticks(ns)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ns, [s["fp_per_case"] for s in sweep], "D-", color="#DD8452", label="nnDet FP/case")
    ax2.axhline(bl_fp_tot / n_cases, color="#DD8452", linestyle="--", alpha=0.5,
                label=f"BL FP/case ({bl_fp_tot/n_cases:.2f})")
    ax2.set_xlabel("Top-N predictions kept per case")
    ax2.set_ylabel("FP per case")
    ax2.set_title("False Positives per Case")
    ax2.set_xticks(ns)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "topn_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved topn_sweep.png")

    # ============================================================
    # Plot 2: Top-1 aggregate comparison bar chart
    # ============================================================
    s1 = sweep[0]  # top-1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Top-1 nnDetection vs Baseline", fontsize=14, fontweight="bold")

    # Metrics
    ax = axes[0]
    labels = ["Precision", "Recall", "F1"]
    bl_v = [bl_prec, bl_rec, bl_f1]
    pp_v = [s1["precision"], s1["recall"], s1["f1"]]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w/2, bl_v, w, label="Baseline", color="#4C72B0")
    b2 = ax.bar(x + w/2, pp_v, w, label="nnDet Top-1", color="#C44E52")
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.set_title("Detection Metrics")
    ax.grid(axis="y", alpha=0.3)

    # Counts
    ax = axes[1]
    x = np.arange(2)
    tp_v = [bl_tp_tot, s1["TP"]]
    fp_v = [bl_fp_tot, s1["FP"]]
    fn_v = [bl_fn_tot, s1["FN"]]
    ax.bar(x, tp_v, 0.5, label="TP", color="#55A868")
    ax.bar(x, fp_v, 0.5, bottom=tp_v, label="FP", color="#C44E52", alpha=0.8)
    ax.bar(x, fn_v, 0.5, bottom=[t+f for t, f in zip(tp_v, fp_v)], label="FN", color="#DD8452", alpha=0.8)
    for i in range(2):
        ax.text(x[i], tp_v[i]/2, f"TP={tp_v[i]}", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        ax.text(x[i], tp_v[i]+fp_v[i]/2, f"FP={fp_v[i]}", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        y_fn = tp_v[i] + fp_v[i] + fn_v[i]/2
        ax.text(x[i], y_fn, f"FN={fn_v[i]}", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "nnDet Top-1"], fontsize=11)
    ax.set_title("Detection Counts")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "top1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved top1_comparison.png")

    # ============================================================
    # Plot 3: Per-case F1 scatter (top-1 vs baseline)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(top1["bl_f1_c"], top1["pp_f1"], s=60, alpha=0.7,
                    edgecolor="black", linewidth=0.5, c=top1["n_gt"], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Equal")
    ax.set_xlabel("Baseline F1", fontsize=12)
    ax.set_ylabel("nnDet Top-1 F1", fontsize=12)
    ax.set_title("Per-Case F1: Baseline vs nnDet Top-1", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="# GT lesions")
    above = (top1["pp_f1"] > top1["bl_f1_c"]).sum()
    below = (top1["pp_f1"] < top1["bl_f1_c"]).sum()
    equal = (top1["pp_f1"] == top1["bl_f1_c"]).sum()
    ax.text(0.05, 0.95, f"Top-1 better: {above}\nBaseline better: {below}\nEqual: {equal}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "top1_f1_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved top1_f1_scatter.png")

    # ============================================================
    # Plot 4: Per-case paired bar for top-1
    # ============================================================
    top1_sorted = top1.sort_values("bl_f1_c", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(n_cases)
    h = 0.4
    ax.barh(y - h/2, top1_sorted["bl_f1_c"], h, label="Baseline", color="#4C72B0", alpha=0.8)
    ax.barh(y + h/2, top1_sorted["pp_f1"], h, label="nnDet Top-1", color="#C44E52", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top1_sorted["stem"], fontsize=6)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Case F1: Baseline vs nnDet Top-1", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "top1_per_case_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved top1_per_case_f1.png")

    # ============================================================
    # Plot 5: Summary dashboard
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("nnDetection Top-N Analysis vs Baseline", fontsize=15, fontweight="bold")

    # (0,0) Top-N F1 sweep
    ax = axes[0, 0]
    ax.plot(ns, [s["f1"] for s in sweep], "^-", color="#C44E52", label="nnDet F1", markersize=6)
    ax.axhline(bl_f1, color="#4C72B0", linestyle="--", linewidth=2, label=f"Baseline F1 ({bl_f1:.3f})")
    ax.set_xlabel("Top-N"); ax.set_ylabel("F1"); ax.set_title("F1 vs Top-N")
    ax.set_xticks(ns); ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (0,1) Top-N Precision sweep
    ax = axes[0, 1]
    ax.plot(ns, [s["precision"] for s in sweep], "o-", color="#4C72B0", label="nnDet Precision", markersize=6)
    ax.axhline(bl_prec, color="#4C72B0", linestyle="--", linewidth=2, label=f"Baseline ({bl_prec:.3f})")
    ax.plot(ns, [s["recall"] for s in sweep], "s-", color="#55A868", label="nnDet Recall", markersize=6)
    ax.axhline(bl_rec, color="#55A868", linestyle="--", linewidth=2, label=f"Baseline ({bl_rec:.3f})")
    ax.set_xlabel("Top-N"); ax.set_ylabel("Score"); ax.set_title("Precision & Recall vs Top-N")
    ax.set_xticks(ns); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # (1,0) Top-1 scatter
    ax = axes[1, 0]
    ax.scatter(top1["bl_f1_c"], top1["pp_f1"], s=40, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Baseline F1"); ax.set_ylabel("nnDet Top-1 F1")
    ax.set_title(f"Per-Case F1 (Top-1 better: {above}, worse: {below})")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    # (1,1) FP/case sweep
    ax = axes[1, 1]
    ax.plot(ns, [s["fp_per_case"] for s in sweep], "D-", color="#DD8452", markersize=6)
    ax.axhline(bl_fp_tot / n_cases, color="#DD8452", linestyle="--", linewidth=2,
               label=f"Baseline ({bl_fp_tot/n_cases:.2f})")
    ax.set_xlabel("Top-N"); ax.set_ylabel("FP/case"); ax.set_title("FP per Case vs Top-N")
    ax.set_xticks(ns); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved summary_dashboard.png")

    # Save CSV
    top1.to_csv(os.path.join(save_dir, "top1_per_case.csv"), index=False)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(os.path.join(save_dir, "topn_sweep.csv"), index=False)
    print(f"\nAll saved to: {save_dir}")


if __name__ == "__main__":
    main()

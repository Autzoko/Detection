"""
Oracle Top-K comparison: for each case, keep the top-K predictions where
K = number of GT lesions in that case. Compare to baseline.
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    if len(pred_boxes) == 0 or n == 0:
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

    save_dir = os.path.join(pp_base, "topk_oracle_comparison")
    os.makedirs(save_dir, exist_ok=True)

    # Load baseline
    baseline_xlsx = "/Users/langtian/Desktop/Prediction_vs_GT_Comparison.xlsx"
    bl_per_file = pd.read_excel(baseline_xlsx, sheet_name="Per-File Summary")

    # Build mapping
    stats_csv = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/dataset_statistics.csv"
    case_mapping = build_case_mapping(stats_csv)
    stem_to_case = {v: k for k, v in case_mapping.items()}

    # ---- Per-case: oracle top-K (K = n_gt) ----
    rows = []
    for _, bl_row in bl_per_file.iterrows():
        bl_fname = bl_row["Filename"]
        stem = bl_fname.replace(".ai", "")
        case_id = stem_to_case.get(stem)
        if case_id is None:
            continue

        # Load predictions
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

        # Oracle top-K
        tp, fp, fn = match_top_n(pred_boxes, pred_scores, gt_boxes, n_gt)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / n_gt if n_gt > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # Also compute top-K for K = 2*n_gt and 3*n_gt
        tp_2k, fp_2k, fn_2k = match_top_n(pred_boxes, pred_scores, gt_boxes, 2 * n_gt)
        p_2k = tp_2k / (tp_2k + fp_2k) if (tp_2k + fp_2k) > 0 else 0
        r_2k = tp_2k / n_gt if n_gt > 0 else 0
        f1_2k = 2 * p_2k * r_2k / (p_2k + r_2k) if (p_2k + r_2k) > 0 else 0

        tp_3k, fp_3k, fn_3k = match_top_n(pred_boxes, pred_scores, gt_boxes, 3 * n_gt)
        p_3k = tp_3k / (tp_3k + fp_3k) if (tp_3k + fp_3k) > 0 else 0
        r_3k = tp_3k / n_gt if n_gt > 0 else 0
        f1_3k = 2 * p_3k * r_3k / (p_3k + r_3k) if (p_3k + r_3k) > 0 else 0

        # Baseline
        bl_tp = bl_row["# TP (Matched)"]
        bl_fp = bl_row["# FP"]
        bl_fn = bl_row["# FN"]
        bl_p = bl_tp / (bl_tp + bl_fp) if (bl_tp + bl_fp) > 0 else 0
        bl_r = bl_tp / n_gt if n_gt > 0 else 0
        bl_f1 = bl_row["Det. F1"] if not pd.isna(bl_row["Det. F1"]) else 0

        rows.append({
            "case_id": case_id, "stem": stem, "n_gt": n_gt,
            # Baseline
            "bl_tp": bl_tp, "bl_fp": bl_fp, "bl_fn": bl_fn,
            "bl_prec": bl_p, "bl_rec": bl_r, "bl_f1": bl_f1,
            "bl_n_pred": bl_row["# Predictions"],
            # Oracle top-K
            "ok_tp": tp, "ok_fp": fp, "ok_fn": fn,
            "ok_prec": p, "ok_rec": r, "ok_f1": f1,
            # Top-2K
            "o2k_tp": tp_2k, "o2k_fp": fp_2k, "o2k_fn": fn_2k,
            "o2k_prec": p_2k, "o2k_rec": r_2k, "o2k_f1": f1_2k,
            # Top-3K
            "o3k_tp": tp_3k, "o3k_fp": fp_3k, "o3k_fn": fn_3k,
            "o3k_prec": p_3k, "o3k_rec": r_3k, "o3k_f1": f1_3k,
        })

    df = pd.DataFrame(rows)
    n_cases = len(df)
    total_gt = df["n_gt"].sum()

    # Aggregate
    def agg(prefix):
        tp = df[f"{prefix}_tp"].sum()
        fp = df[f"{prefix}_fp"].sum()
        fn = df[f"{prefix}_fn"].sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / total_gt if total_gt > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r,
                "f1": f1, "fp_per_case": fp / n_cases}

    bl_agg = agg("bl")
    ok_agg = agg("ok")
    o2k_agg = agg("o2k")
    o3k_agg = agg("o3k")

    # Print table
    print(f"\n{'=' * 85}")
    print(f"  ORACLE TOP-K COMPARISON  ({n_cases} cases, {total_gt} GT)")
    print(f"  K = number of GT lesions per case (oracle knowledge)")
    print(f"{'=' * 85}")
    print(f"{'Method':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'FP/case':>10}")
    print("-" * 85)
    for label, m in [("Baseline", bl_agg), ("nnDet Top-K", ok_agg),
                     ("nnDet Top-2K", o2k_agg), ("nnDet Top-3K", o3k_agg)]:
        print(f"{label:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['fp_per_case']:>10.2f}")
    print(f"{'=' * 85}")

    # Per-case table
    print(f"\n{'=' * 110}")
    print(f"  PER-CASE TABLE (Oracle Top-K)")
    print(f"{'=' * 110}")
    print(f"{'Case':<15} {'GT':>3} {'BL_P':>6} {'BL_R':>6} {'BL_F1':>6} {'BL_TP':>5} {'BL_FP':>5} "
          f"{'OK_P':>6} {'OK_R':>6} {'OK_F1':>6} {'OK_TP':>5} {'OK_FP':>5} {'Winner':>10}")
    print("-" * 110)
    ok_wins = 0; bl_wins = 0; ties = 0
    for _, r in df.iterrows():
        if r["ok_f1"] > r["bl_f1"]:
            winner = "Top-K"; ok_wins += 1
        elif r["ok_f1"] < r["bl_f1"]:
            winner = "Baseline"; bl_wins += 1
        else:
            winner = "Tie"; ties += 1
        print(f"{r['stem']:<15} {r['n_gt']:>3} "
              f"{r['bl_prec']:>6.3f} {r['bl_rec']:>6.3f} {r['bl_f1']:>6.3f} "
              f"{r['bl_tp']:>5} {r['bl_fp']:>5} "
              f"{r['ok_prec']:>6.3f} {r['ok_rec']:>6.3f} {r['ok_f1']:>6.3f} "
              f"{r['ok_tp']:>5} {r['ok_fp']:>5} "
              f"{winner:>10}")
    print(f"{'=' * 110}")
    print(f"Top-K wins: {ok_wins}, Baseline wins: {bl_wins}, Ties: {ties}")

    # ============================================================
    # Plot 1: Aggregate comparison bar chart
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Oracle Top-K (K = #GT) vs Baseline", fontsize=14, fontweight="bold")

    ax = axes[0]
    methods = ["Baseline", "Top-K", "Top-2K", "Top-3K"]
    x = np.arange(3)
    w = 0.2
    colors = ["#4C72B0", "#C44E52", "#DD8452", "#55A868"]
    for i, (label, m) in enumerate([("Baseline", bl_agg), ("Top-K", ok_agg),
                                     ("Top-2K", o2k_agg), ("Top-3K", o3k_agg)]):
        vals = [m["precision"], m["recall"], m["f1"]]
        bars = ax.bar(x + i * w - 1.5 * w, vals, w, label=label, color=colors[i])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.set_title("Detection Metrics")
    ax.grid(axis="y", alpha=0.3)

    # Counts
    ax = axes[1]
    x = np.arange(4)
    tp_v = [bl_agg["TP"], ok_agg["TP"], o2k_agg["TP"], o3k_agg["TP"]]
    fp_v = [bl_agg["FP"], ok_agg["FP"], o2k_agg["FP"], o3k_agg["FP"]]
    fn_v = [bl_agg["FN"], ok_agg["FN"], o2k_agg["FN"], o3k_agg["FN"]]
    ax.bar(x, tp_v, 0.5, label="TP", color="#55A868")
    ax.bar(x, fp_v, 0.5, bottom=tp_v, label="FP", color="#C44E52", alpha=0.8)
    ax.bar(x, fn_v, 0.5, bottom=[t + f for t, f in zip(tp_v, fp_v)],
           label="FN", color="#DD8452", alpha=0.8)
    for i in range(4):
        ax.text(x[i], tp_v[i] / 2, f"{tp_v[i]}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        if fp_v[i] > 3:
            ax.text(x[i], tp_v[i] + fp_v[i] / 2, f"{fp_v[i]}", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        if fn_v[i] > 3:
            ax.text(x[i], tp_v[i] + fp_v[i] + fn_v[i] / 2, f"{fn_v[i]}", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_title("TP / FP / FN Counts")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "aggregate_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved aggregate_comparison.png")

    # ============================================================
    # Plot 2: Per-case F1 scatter (oracle top-K vs baseline)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["bl_f1"], df["ok_f1"], s=60, alpha=0.7,
                    edgecolor="black", linewidth=0.5, c=df["n_gt"], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Equal")
    ax.set_xlabel("Baseline F1", fontsize=12)
    ax.set_ylabel("nnDet Oracle Top-K F1", fontsize=12)
    ax.set_title("Per-Case F1: Oracle Top-K vs Baseline", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="# GT lesions (= K)")
    ax.text(0.05, 0.95, f"Top-K better: {ok_wins}\nBaseline better: {bl_wins}\nEqual: {ties}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "f1_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved f1_scatter.png")

    # ============================================================
    # Plot 3: Per-case paired bar (sorted by baseline F1)
    # ============================================================
    df_sorted = df.sort_values("bl_f1", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(n_cases)
    h = 0.4
    ax.barh(y - h / 2, df_sorted["bl_f1"], h, label="Baseline", color="#4C72B0", alpha=0.8)
    ax.barh(y + h / 2, df_sorted["ok_f1"], h, label="nnDet Top-K", color="#C44E52", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['stem']} (K={r['n_gt']})" for _, r in df_sorted.iterrows()], fontsize=6)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Case F1: Baseline vs nnDet Oracle Top-K (K=#GT)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_case_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved per_case_f1.png")

    # ============================================================
    # Plot 4: Summary dashboard
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Oracle Top-K Analysis (K = #GT lesions per case)",
                 fontsize=15, fontweight="bold")

    # (0,0) Aggregate P/R/F1
    ax = axes[0, 0]
    x = np.arange(3)
    w = 0.3
    for i, (label, m, c) in enumerate([("Baseline", bl_agg, "#4C72B0"),
                                        ("Top-K", ok_agg, "#C44E52"),
                                        ("Top-2K", o2k_agg, "#DD8452")]):
        vals = [m["precision"], m["recall"], m["f1"]]
        bars = ax.bar(x + i * w - w, vals, w, label=label, color=c)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.set_title("Aggregate Metrics")
    ax.grid(axis="y", alpha=0.3)

    # (0,1) FP/case comparison
    ax = axes[0, 1]
    methods = ["Baseline", "Top-K", "Top-2K", "Top-3K"]
    fp_cases = [bl_agg["fp_per_case"], ok_agg["fp_per_case"],
                o2k_agg["fp_per_case"], o3k_agg["fp_per_case"]]
    bars = ax.bar(methods, fp_cases, color=colors)
    for bar, v in zip(bars, fp_cases):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("FP per case")
    ax.set_title("False Positives per Case")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) F1 scatter
    ax = axes[1, 0]
    sc = ax.scatter(df["bl_f1"], df["ok_f1"], s=40, alpha=0.7,
                    edgecolor="black", linewidth=0.5, c=df["n_gt"], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Baseline F1"); ax.set_ylabel("Top-K F1")
    ax.set_title(f"Per-Case F1 (Top-K better: {ok_wins}, worse: {bl_wins})")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="K (#GT)")

    # (1,1) Recall by #GT group
    ax = axes[1, 1]
    gt_groups = sorted(df["n_gt"].unique())
    bl_rec_by_gt = []
    ok_rec_by_gt = []
    for g in gt_groups:
        sub = df[df["n_gt"] == g]
        bl_r = sub["bl_tp"].sum() / sub["n_gt"].sum() if sub["n_gt"].sum() > 0 else 0
        ok_r = sub["ok_tp"].sum() / sub["n_gt"].sum() if sub["n_gt"].sum() > 0 else 0
        bl_rec_by_gt.append(bl_r)
        ok_rec_by_gt.append(ok_r)
    x = np.arange(len(gt_groups))
    w = 0.35
    ax.bar(x - w / 2, bl_rec_by_gt, w, label="Baseline", color="#4C72B0")
    ax.bar(x + w / 2, ok_rec_by_gt, w, label="Top-K", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gt_groups])
    ax.set_xlabel("# GT lesions (K)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall by Number of GT Lesions")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved summary_dashboard.png")

    # Save CSV
    df.to_csv(os.path.join(save_dir, "per_case_comparison.csv"), index=False)
    print(f"\nAll saved to: {save_dir}")


if __name__ == "__main__":
    main()

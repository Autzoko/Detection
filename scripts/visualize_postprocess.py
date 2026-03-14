"""
Visualize post-processing pipeline results.

Produces:
  1. Step-by-step comparison bar charts (Precision, Recall, F1, FP/case)
  2. Precision-Recall curve from threshold sweep
  3. FP/case vs Recall trade-off curve
  4. Threshold sweep line plots (P, R, F1 vs threshold)
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/postprocessed_pipeline"


def load_comparison(path):
    steps, metrics = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(row["Step"])
            metrics.append({
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "fp_per_case": float(row["FP_per_case"]),
                "TP": int(row["TP"]),
                "FP": int(row["FP"]),
                "FN": int(row["FN"]),
            })
    return steps, metrics


def load_sweep(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "threshold": float(row["Threshold"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "fp_per_case": float(row["FP_per_case"]),
                "TP": int(row["TP"]),
                "FP": int(row["FP"]),
            })
    return rows


def plot_comparison(steps, metrics, save_dir):
    """Bar charts showing metrics at each pipeline step."""
    # Shorten step labels
    short_labels = []
    for s in steps:
        s = s.replace("+ ", "").replace("Baseline", "Baseline")
        short_labels.append(s)

    x = np.arange(len(short_labels))
    width = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Post-Processing Pipeline: Step-by-Step Impact", fontsize=14, fontweight="bold")

    # Precision
    ax = axes[0, 0]
    vals = [m["precision"] for m in metrics]
    bars = ax.bar(x, vals, width, color="#4C72B0", edgecolor="white")
    ax.set_ylabel("Precision")
    ax.set_title("Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # Recall
    ax = axes[0, 1]
    vals = [m["recall"] for m in metrics]
    bars = ax.bar(x, vals, width, color="#55A868", edgecolor="white")
    ax.set_ylabel("Recall")
    ax.set_title("Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # F1
    ax = axes[1, 0]
    vals = [m["f1"] for m in metrics]
    bars = ax.bar(x, vals, width, color="#C44E52", edgecolor="white")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # FP per case
    ax = axes[1, 1]
    vals = [m["fp_per_case"] for m in metrics]
    bars = ax.bar(x, vals, width, color="#DD8452", edgecolor="white")
    ax.set_ylabel("FP / case")
    ax.set_title("False Positives per Case")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "pipeline_steps_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_threshold_sweep(sweep, save_dir):
    """Line plots of P, R, F1 vs threshold."""
    thresholds = [r["threshold"] for r in sweep]
    precision = [r["precision"] for r in sweep]
    recall = [r["recall"] for r in sweep]
    f1 = [r["f1"] for r in sweep]
    fp_case = [r["fp_per_case"] for r in sweep]

    # Best F1
    best_idx = np.argmax(f1)
    best_t = thresholds[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Score Threshold Sweep (after Anatomy Mask + 50mm Clustering)",
                 fontsize=13, fontweight="bold")

    # P, R, F1 vs threshold
    ax1.plot(thresholds, precision, "o-", color="#4C72B0", label="Precision", markersize=4)
    ax1.plot(thresholds, recall, "s-", color="#55A868", label="Recall", markersize=4)
    ax1.plot(thresholds, f1, "^-", color="#C44E52", label="F1", markersize=4)
    ax1.axvline(best_t, color="gray", linestyle="--", alpha=0.7, label=f"Best F1 @ {best_t}")
    ax1.set_xlabel("Score Threshold")
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Precision / Recall / F1")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # FP/case vs threshold
    ax2.plot(thresholds, fp_case, "D-", color="#DD8452", markersize=4)
    ax2.axvline(best_t, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Score Threshold")
    ax2.set_ylabel("FP per Case")
    ax2.set_title("False Positives per Case")
    ax2.grid(True, alpha=0.3)

    # Annotate best point
    ax2.annotate(f"Best F1: {best_t}\n{fp_case[best_idx]:.1f} FP/case",
                 xy=(best_t, fp_case[best_idx]),
                 xytext=(best_t - 0.15, fp_case[best_idx] + 10),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(save_dir, "threshold_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_pr_curve(sweep, save_dir):
    """Precision-Recall curve from threshold sweep."""
    recall = [r["recall"] for r in sweep]
    precision = [r["precision"] for r in sweep]
    thresholds = [r["threshold"] for r in sweep]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, "o-", color="#4C72B0", markersize=5, linewidth=2)

    # Annotate select thresholds
    for i, t in enumerate(thresholds):
        if t in [0.1, 0.3, 0.5, 0.7, 0.85, 0.9]:
            ax.annotate(f"t={t}", xy=(recall[i], precision[i]),
                        xytext=(recall[i] + 0.02, precision[i] + 0.005),
                        fontsize=8, color="gray")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (Post-Processed)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "precision_recall_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_fp_vs_recall(sweep, save_dir):
    """FP/case vs Recall trade-off (FROC-style)."""
    recall = [r["recall"] for r in sweep]
    fp_case = [r["fp_per_case"] for r in sweep]
    thresholds = [r["threshold"] for r in sweep]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fp_case, recall, "s-", color="#C44E52", markersize=5, linewidth=2)

    # Annotate select thresholds
    for i, t in enumerate(thresholds):
        if t in [0.1, 0.3, 0.5, 0.7, 0.85, 0.9]:
            ax.annotate(f"t={t}", xy=(fp_case[i], recall[i]),
                        xytext=(fp_case[i] + 1, recall[i] - 0.02),
                        fontsize=8, color="gray")

    ax.set_xlabel("False Positives per Case", fontsize=12)
    ax.set_ylabel("Recall (Sensitivity)", fontsize=12)
    ax.set_title("FROC-style: Recall vs FP/case (Post-Processed)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fp_vs_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_tp_fp_stacked(steps, metrics, save_dir):
    """Stacked bar chart showing TP vs FP counts at each step."""
    short_labels = [s.replace("+ ", "") for s in steps]
    x = np.arange(len(short_labels))

    tp_vals = [m["TP"] for m in metrics]
    fp_vals = [m["FP"] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, tp_vals, 0.6, label="TP", color="#55A868", edgecolor="white")
    ax.bar(x, fp_vals, 0.6, bottom=tp_vals, label="FP", color="#C44E52", alpha=0.7, edgecolor="white")

    for i in range(len(x)):
        ax.text(x[i], tp_vals[i] / 2, f"TP={tp_vals[i]}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        if fp_vals[i] > 200:
            ax.text(x[i], tp_vals[i] + fp_vals[i] / 2, f"FP={fp_vals[i]}", ha="center",
                    va="center", fontsize=8, color="white")
        else:
            ax.text(x[i], tp_vals[i] + fp_vals[i] + 50, f"FP={fp_vals[i]}", ha="center",
                    va="bottom", fontsize=8, color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("TP and FP Counts at Each Pipeline Step", fontsize=13, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "tp_fp_stacked.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    comp_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    sweep_path = os.path.join(OUTPUT_DIR, "threshold_sweep.csv")

    steps, metrics = load_comparison(comp_path)
    sweep = load_sweep(sweep_path)

    plot_comparison(steps, metrics, OUTPUT_DIR)
    plot_threshold_sweep(sweep, OUTPUT_DIR)
    plot_pr_curve(sweep, OUTPUT_DIR)
    plot_fp_vs_recall(sweep, OUTPUT_DIR)
    plot_tp_fp_stacked(steps, metrics, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

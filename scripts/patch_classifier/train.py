"""
Train 3D patch classifier for nnDetection FP reduction.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --device cuda:0
    python train.py --config config.yaml --resume checkpoint.pt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from model import PatchClassifier3D, count_parameters
from dataset import PatchDataset


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Cosine decay with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        patches = batch["patch"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * patches.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += patches.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []

    for batch in dataloader:
        patches = batch["patch"].to(device)
        labels = batch["label"].to(device)

        logits = model(patches)
        loss = criterion(logits, labels)

        total_loss += loss.item() * patches.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += patches.size(0)

        all_probs.extend(probs.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # Compute precision, recall, F1 at threshold 0.5
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    pred_pos = all_probs >= 0.5
    tp = ((pred_pos) & (all_labels == 1)).sum()
    fp = ((pred_pos) & (all_labels == 0)).sum()
    fn = ((~pred_pos) & (all_labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probs": all_probs,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Datasets
    dataset_dir = paths["dataset_dir"]
    aug_cfg = train_cfg["augmentation"]

    train_dataset = PatchDataset(dataset_dir, split="train", augment=True, aug_cfg=aug_cfg)
    val_dataset = PatchDataset(dataset_dir, split="val", augment=False)

    n_pos_train, n_neg_train = train_dataset.get_label_counts()
    n_pos_val, n_neg_val = val_dataset.get_label_counts()
    print(f"Train: {len(train_dataset)} samples ({n_pos_train} pos, {n_neg_train} neg)")
    print(f"Val:   {len(val_dataset)} samples ({n_pos_val} pos, {n_neg_val} neg)")

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = PatchClassifier3D(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        num_blocks=model_cfg["num_blocks"],
        fc_hidden=model_cfg["fc_hidden"],
        dropout=model_cfg["dropout"],
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss with positive class weight
    pos_weight_val = train_cfg["pos_weight"]
    if pos_weight_val == "auto":
        pos_weight_val = train_dataset.get_pos_weight()
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"Positive class weight: {pos_weight_val:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg["warmup_epochs"], train_cfg["epochs"])

    # Resume
    start_epoch = 0
    best_f1 = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint.get("best_f1", 0)
        print(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    # Checkpoint directory
    ckpt_dir = Path(paths["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    patience_counter = 0
    history = []

    print(f"\n{'='*70}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val P':>5} | "
          f"{'Val R':>5} | {'Val F1':>6} | {'LR':>8}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:9.4f} | "
              f"{val_metrics['loss']:8.4f} | {val_metrics['accuracy']:7.4f} | "
              f"{val_metrics['precision']:5.3f} | {val_metrics['recall']:5.3f} | "
              f"{val_metrics['f1']:6.4f} | {current_lr:8.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": current_lr,
        })

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": cfg,
            }, ckpt_dir / "best.pt")
            print(f"  → New best F1={best_f1:.4f}, saved to {ckpt_dir / 'best.pt'}")
        else:
            patience_counter += 1

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "config": cfg,
        }, ckpt_dir / "last.pt")

        # Early stopping
        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={train_cfg['early_stopping_patience']})")
            break

    # Save training history
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()

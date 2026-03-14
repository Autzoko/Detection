"""
PyTorch Dataset for 3D patch classifier.

Loads pre-extracted .npy patches and their labels from manifest.json.
Applies augmentation during training: random flips, rotation, intensity jitter.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import rotate
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Dataset of 3D patches for binary classification."""

    def __init__(self, dataset_dir, split="train", augment=False, aug_cfg=None):
        """
        Args:
            dataset_dir: root dataset directory containing train/ and val/ subdirs
            split: "train" or "val"
            augment: whether to apply data augmentation
            aug_cfg: augmentation config dict with keys:
                flip_prob, rotation_degrees, intensity_jitter_prob, intensity_jitter_range
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

        # Load manifest
        manifest_path = self.dataset_dir / split / "manifest.json"
        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.patch_dir = self.dataset_dir / split / "patches"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample["name"]
        label = sample["label"]

        # Load patch
        patch = np.load(self.patch_dir / f"{name}.npy").astype(np.float32)

        # Apply augmentation
        if self.augment:
            patch = self._augment(patch)

        # Add channel dimension: (D, H, W) → (1, D, H, W)
        patch = patch[np.newaxis, ...]
        patch_tensor = torch.from_numpy(patch)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return {
            "patch": patch_tensor,
            "label": label_tensor,
            "name": name,
            "case_id": sample["case_id"],
            "pred_score": sample.get("pred_score", 0.0),
        }

    def _augment(self, patch):
        """Apply random augmentations to a 3D patch."""
        flip_prob = self.aug_cfg.get("flip_prob", 0.5)
        rot_deg = self.aug_cfg.get("rotation_degrees", 15)
        jitter_prob = self.aug_cfg.get("intensity_jitter_prob", 0.3)
        jitter_range = self.aug_cfg.get("intensity_jitter_range", 0.1)

        # Random flip along each axis
        for axis in range(3):
            if random.random() < flip_prob:
                patch = np.flip(patch, axis=axis).copy()

        # Random rotation (around z-axis for simplicity, avoids heavy interpolation)
        if rot_deg > 0:
            angle = random.uniform(-rot_deg, rot_deg)
            # Rotate in the y-x plane (axes 1, 2)
            patch = rotate(patch, angle, axes=(1, 2), reshape=False,
                           order=1, mode="constant", cval=0.0)

        # Intensity jitter
        if random.random() < jitter_prob:
            # Multiplicative brightness change
            factor = 1.0 + random.uniform(-jitter_range, jitter_range)
            patch = patch * factor
            # Additive shift
            shift = random.uniform(-jitter_range * 0.5, jitter_range * 0.5)
            patch = patch + shift
            # Re-clip to [0, 1]
            patch = np.clip(patch, 0.0, 1.0)

        return patch.astype(np.float32)

    def get_pos_weight(self):
        """Compute positive class weight from label distribution."""
        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        n_neg = sum(1 for s in self.samples if s["label"] == 0)
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def get_label_counts(self):
        """Return (n_positive, n_negative)."""
        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        n_neg = len(self.samples) - n_pos
        return n_pos, n_neg

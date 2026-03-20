"""
Patch-level binary classification head for nnDetection.

Predicts whether a patch contains a lesion (positive) or not (negative).
Uses global average pooling on FPN features followed by an MLP.
Can be used to reduce false positives by providing a case-level confidence.
"""

import torch
import torch.nn as nn

from typing import Dict, List, Optional, Sequence

from nndet.arch.heads.comb import AbstractHead


class PatchClassifier(AbstractHead):
    """
    Patch-level binary classifier that operates on FPN features.

    Architecture:
        FPN features → Global Average Pooling → Concat → MLP → Sigmoid

    The classifier takes multi-scale FPN features, pools each level,
    concatenates them, and predicts a binary positive/negative label.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        hidden_dim: int = 128,
        dropout: float = 0.3,
        loss_weight: float = 1.0,
    ):
        """
        Args:
            in_channels: number of channels at each FPN level
            decoder_levels: which decoder levels are used
            hidden_dim: hidden dimension of MLP
            dropout: dropout rate
            loss_weight: weight for the classification loss
        """
        super().__init__()
        self.decoder_levels = decoder_levels
        self.loss_weight = loss_weight

        # Total input features = sum of channels from all FPN levels used
        total_channels = sum(in_channels[i] for i in decoder_levels)

        self.classifier = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        feature_maps: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            feature_maps: FPN features at decoder_levels.
                Each tensor has shape [N, C, *spatial_dims]

        Returns:
            Dict with 'patch_logits': [N, 1] logits
        """
        pooled = []
        for fm in feature_maps:
            # Global average pooling over spatial dimensions
            dims = list(range(2, fm.ndim))  # spatial dims
            pooled.append(fm.mean(dim=dims))

        # Concatenate pooled features from all levels
        x = torch.cat(pooled, dim=1)  # [N, total_channels]
        logits = self.classifier(x)  # [N, 1]

        return {"patch_logits": logits}

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute binary classification loss.

        A patch is positive if it contains at least one GT box.

        Args:
            pred: predictions with 'patch_logits' [N, 1]
            target_boxes: list of GT boxes per image in batch

        Returns:
            Dict with 'patch_cls' loss
        """
        logits = pred["patch_logits"]  # [N, 1]
        device = logits.device

        # Create binary labels: 1 if patch has GT boxes, 0 otherwise
        labels = torch.tensor(
            [1.0 if len(boxes) > 0 else 0.0 for boxes in target_boxes],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)  # [N, 1]

        loss = self.loss_weight * self.loss_fn(logits, labels)
        return {"patch_cls": loss}

    def postprocess_for_inference(
        self,
        pred: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert logits to probabilities for inference.

        Args:
            pred: predictions with 'patch_logits'

        Returns:
            Dict with 'pred_patch_cls': probability of positive [N, 1]
        """
        return {"pred_patch_cls": torch.sigmoid(pred["patch_logits"])}

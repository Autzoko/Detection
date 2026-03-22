"""
BI-RADS classification head for nnDetection.

Operates on FPN features (like PatchClassifier) but predicts BI-RADS class
for each detected lesion instead of binary positive/negative.

Designed to be added alongside the existing PatchClassifier without
modifying existing code.
"""

import torch
import torch.nn as nn

from typing import Dict, List, Sequence

from nndet.arch.heads.comb import AbstractHead


class BiRadsClassifier(AbstractHead):
    """
    Patch-level BI-RADS classifier that operates on FPN features.

    Architecture:
        FPN features -> Global Average Pooling -> Concat -> MLP -> num_classes logits

    Predicts BI-RADS class (e.g., 2/3/4) for patches that contain lesions.
    Only computes loss on positive patches (those with GT boxes).
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        loss_weight: float = 1.0,
    ):
        """
        Args:
            in_channels: number of channels at each FPN level
            decoder_levels: which decoder levels are used
            num_classes: number of BI-RADS classes (e.g., 3 for BIRADS 2/3/4)
            hidden_dim: hidden dimension of MLP
            dropout: dropout rate
            loss_weight: weight for the classification loss
        """
        super().__init__()
        self.decoder_levels = decoder_levels
        self.num_classes = num_classes
        self.loss_weight = loss_weight

        # Total input features = sum of channels from all FPN levels used
        total_channels = sum(in_channels[i] for i in decoder_levels)

        self.classifier = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

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
            Dict with 'birads_logits': [N, num_classes] logits
        """
        pooled = []
        for fm in feature_maps:
            dims = list(range(2, fm.ndim))  # spatial dims
            pooled.append(fm.mean(dim=dims))

        x = torch.cat(pooled, dim=1)  # [N, total_channels]
        logits = self.classifier(x)   # [N, num_classes]

        return {"birads_logits": logits}

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BI-RADS classification loss on positive patches only.

        For patches with multiple GT boxes, uses the class of the first box
        (majority class could be used as an alternative).

        Args:
            pred: predictions with 'birads_logits' [N, num_classes]
            target_boxes: list of GT boxes per image in batch
            target_classes: list of GT class labels per image in batch

        Returns:
            Dict with 'birads_cls' loss (zero if no positive patches)
        """
        logits = pred["birads_logits"]  # [N, num_classes]
        device = logits.device

        # Collect logits and labels for positive patches only
        pos_logits = []
        pos_labels = []

        for i, (boxes, classes) in enumerate(zip(target_boxes, target_classes)):
            if len(boxes) > 0 and len(classes) > 0:
                pos_logits.append(logits[i])
                # Use the class of the first GT box in this patch
                pos_labels.append(classes[0].long())

        if len(pos_logits) == 0:
            # No positive patches in this batch, return zero loss
            return {"birads_cls": torch.tensor(0.0, device=device, requires_grad=True)}

        pos_logits = torch.stack(pos_logits)   # [P, num_classes]
        pos_labels = torch.stack(pos_labels)   # [P]

        loss = self.loss_weight * self.loss_fn(pos_logits, pos_labels)
        return {"birads_cls": loss}

    def postprocess_for_inference(
        self,
        pred: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert logits to class probabilities and predictions.

        Args:
            pred: predictions with 'birads_logits'

        Returns:
            Dict with:
                'pred_birads_probs': [N, num_classes] softmax probabilities
                'pred_birads_labels': [N] predicted class indices
        """
        probs = torch.softmax(pred["birads_logits"], dim=1)
        labels = probs.argmax(dim=1)
        return {
            "pred_birads_probs": probs,
            "pred_birads_labels": labels,
        }

"""
BI-RADS classification head for nnDetection.

Uses RoI-based feature extraction: for each detected/GT box, crops
multi-scale FPN features at the box location, adaptive-pools to a
fixed size, then classifies with an MLP.

This replaces the previous GAP-based approach which pooled over the
entire patch and lost lesion-specific spatial information.
"""

import torch
import torch.nn as nn

from typing import Dict, List, Sequence

from nndet.arch.heads.comb import AbstractHead


class BiRadsClassifier(AbstractHead):
    """
    RoI-based BI-RADS classifier on FPN features.

    Architecture per box:
        FPN features → crop at box coords → AdaptiveAvgPool3d(roi_size)
        → concat across FPN levels → MLP → num_classes logits
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_levels: Sequence[int],
        num_classes: int = 3,
        roi_size: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        loss_weight: float = 1.0,
    ):
        """
        Args:
            in_channels: channels at each FPN level
            decoder_levels: which decoder levels are used
            num_classes: number of BI-RADS classes (e.g., 3 for 2/3/4)
            roi_size: spatial size after adaptive pooling per level
            hidden_dim: MLP hidden dimension
            dropout: dropout rate
            loss_weight: weight for classification loss
        """
        super().__init__()
        self.decoder_levels = decoder_levels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.roi_size = roi_size

        total_channels = sum(in_channels[i] for i in decoder_levels)
        self.feature_dim = total_channels * roi_size ** 3

        self.pool = nn.AdaptiveAvgPool3d(roi_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def _extract_roi_features(
        self,
        feature_maps: List[torch.Tensor],
        boxes: torch.Tensor,
        input_shape: Sequence[int],
    ) -> torch.Tensor:
        """
        Extract multi-scale RoI features for each box.

        Args:
            feature_maps: FPN features, each [1, C, *spatial] (single image)
            boxes: [N, 6] in input coords [min_d0, min_d1, min_d2, max_d0, max_d1, max_d2]
            input_shape: spatial shape of network input (d0, d1, d2)

        Returns:
            [N, feature_dim] concatenated RoI features
        """
        if len(boxes) == 0:
            return torch.zeros(0, self.feature_dim, device=feature_maps[0].device)

        all_features = []
        for box in boxes:
            level_feats = []
            for fm in feature_maps:
                fm_shape = fm.shape[2:]  # spatial dims
                scales = [float(fs) / float(ins) for fs, ins in zip(fm_shape, input_shape)]

                # Map box to feature map coordinates
                coords_min = []
                coords_max = []
                for d in range(3):
                    lo = int(box[d].item() * scales[d])
                    hi = int(box[d + 3].item() * scales[d]) + 1
                    # Clamp to valid range
                    lo = max(0, min(lo, fm_shape[d] - 1))
                    hi = max(lo + 1, min(hi, fm_shape[d]))
                    coords_min.append(lo)
                    coords_max.append(hi)

                roi = fm[0, :,
                         coords_min[0]:coords_max[0],
                         coords_min[1]:coords_max[1],
                         coords_min[2]:coords_max[2]]
                pooled = self.pool(roi.unsqueeze(0))  # [1, C, rs, rs, rs]
                level_feats.append(pooled.flatten())

            all_features.append(torch.cat(level_feats))

        return torch.stack(all_features)

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        boxes_per_image: List[torch.Tensor],
        input_shape: Sequence[int],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Predict BI-RADS class for each detected box.

        Args:
            feature_maps: FPN features [B, C, *spatial]
            boxes_per_image: list of [N_i, 6] boxes per image in batch
            input_shape: spatial shape of input

        Returns:
            Dict with per-image lists:
                'birads_probs': List[[N_i, num_classes]]
                'birads_labels': List[[N_i]]
        """
        all_probs = []
        all_labels = []

        for img_idx, boxes in enumerate(boxes_per_image):
            device = feature_maps[0].device
            if len(boxes) == 0:
                all_probs.append(torch.zeros(0, self.num_classes, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            fm_single = [fm[img_idx:img_idx + 1] for fm in feature_maps]
            features = self._extract_roi_features(fm_single, boxes, input_shape)
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs)
            all_labels.append(probs.argmax(dim=1))

        return {"birads_probs": all_probs, "birads_labels": all_labels}

    def compute_loss(
        self,
        feature_maps: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
        input_shape: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BI-RADS loss using GT boxes for RoI extraction.

        Each GT box gets its own RoI features and classification target.

        Args:
            feature_maps: FPN features [B, C, *spatial]
            target_boxes: list of [N_i, 6] GT boxes per image
            target_classes: list of [N_i] GT classes per image
            input_shape: spatial shape of input

        Returns:
            Dict with 'birads_cls' loss
        """
        device = feature_maps[0].device
        all_features = []
        all_labels = []

        for img_idx, (boxes, classes) in enumerate(zip(target_boxes, target_classes)):
            if len(boxes) == 0 or len(classes) == 0:
                continue
            fm_single = [fm[img_idx:img_idx + 1] for fm in feature_maps]
            features = self._extract_roi_features(fm_single, boxes, input_shape)
            all_features.append(features)
            all_labels.append(classes.long().to(device))

        if not all_features:
            return {"birads_cls": torch.tensor(0.0, device=device, requires_grad=True)}

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Clamp labels to valid range to prevent NaN from CrossEntropyLoss
        all_labels = all_labels.clamp(0, self.num_classes - 1)

        logits = self.classifier(all_features)
        loss = self.loss_weight * self.loss_fn(logits, all_labels)

        # Guard against NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            return {"birads_cls": torch.tensor(0.0, device=device, requires_grad=True)}

        return {"birads_cls": loss}

    def postprocess_for_inference(self, pred):
        """Backward compatibility stub (not used in RoI mode)."""
        return pred

"""
Extended RetinaNet with BI-RADS classification head.

Subclasses BaseRetinaNet to add a BI-RADS classifier alongside
the existing detection head and PatchClassifier. Does not modify
the original retina.py.

The BI-RADS classifier uses RoI-based feature extraction:
  - Training: extracts features at GT box locations
  - Inference: extracts features at predicted box locations
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple, Dict, Optional, Union

from nndet.core.retina import BaseRetinaNet
from nndet.arch.encoder.abstract import EncoderType
from nndet.arch.decoder.base import DecoderType
from nndet.arch.heads.segmenter import SegmenterType
from nndet.arch.heads.comb import HeadType
from nndet.arch.heads.patch_classifier import PatchClassifier
from nndet.arch.heads.birads_classifier import BiRadsClassifier
from nndet.core import boxes as box_utils
from nndet.core.boxes.anchors import AnchorGeneratorType


class BiRadsRetinaNet(BaseRetinaNet):
    """
    RetinaNet extended with a BI-RADS classification head.

    Inherits all detection, segmentation, and patch classification
    functionality from BaseRetinaNet. Adds:
      - birads_classifier: RoI-based BI-RADS class prediction per box
      - birads_cls loss during training (using GT boxes)
      - pred_birads_probs / pred_birads_labels during inference (using pred boxes)
    """

    def __init__(
        self,
        dim: int,
        encoder: EncoderType,
        decoder: DecoderType,
        head: HeadType,
        num_classes: int,
        anchor_generator: AnchorGeneratorType,
        matcher: box_utils.MatcherType,
        decoder_levels: tuple = (2, 3, 4, 5),
        score_thresh: float = None,
        detections_per_img: int = 100,
        topk_candidates: int = 10000,
        remove_small_boxes: float = 1e-2,
        nms_thresh: float = 0.9,
        segmenter: Optional[SegmenterType] = None,
        patch_classifier: Optional[PatchClassifier] = None,
        birads_classifier: Optional[BiRadsClassifier] = None,
    ):
        super().__init__(
            dim=dim,
            encoder=encoder,
            decoder=decoder,
            head=head,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            matcher=matcher,
            decoder_levels=decoder_levels,
            score_thresh=score_thresh,
            detections_per_img=detections_per_img,
            topk_candidates=topk_candidates,
            remove_small_boxes=remove_small_boxes,
            nms_thresh=nms_thresh,
            segmenter=segmenter,
            patch_classifier=patch_classifier,
        )
        self.birads_classifier = birads_classifier

    def forward(
        self,
        inp: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor],
               Dict[str, torch.Tensor],
               Optional[Dict[str, torch.Tensor]],
               List[torch.Tensor]]:
        """
        Forward pass. Returns FPN feature maps (instead of birads predictions)
        so that birads can be computed later with boxes.

        Returns:
            pred_detection, anchors, pred_seg, pred_patch_cls, feature_maps_head
        """
        features_maps_all = self.decoder(self.encoder(inp))
        feature_maps_head = [features_maps_all[i] for i in self.decoder_levels]

        pred_detection = self.head(feature_maps_head)
        anchors = self.anchor_generator(inp, feature_maps_head)

        pred_seg = self.segmenter(features_maps_all) if self.segmenter is not None else None
        pred_patch_cls = self.patch_classifier(feature_maps_head) if self.patch_classifier is not None else None

        # Return feature maps for RoI-based birads (computed later with boxes)
        return pred_detection, anchors, pred_seg, pred_patch_cls, feature_maps_head

    def train_step(
        self,
        images: Tensor,
        targets: dict,
        evaluation: bool,
        batch_num: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """
        Training step with BI-RADS classification loss.

        The detection head uses binary classes (all foreground = class 0)
        while the BI-RADS classifier uses the original multi-class labels
        with RoI features extracted at GT box locations.
        """
        target_boxes: List[Tensor] = targets["target_boxes"]
        target_classes: List[Tensor] = targets["target_classes"]
        target_seg: Tensor = targets["target_seg"]

        pred_detection, anchors, pred_seg, pred_patch_cls, feature_maps_head = self(images)

        # For the detection head: map all foreground classes to 0 (binary)
        binary_classes = [torch.zeros_like(tc) for tc in target_classes]

        labels, matched_gt_boxes = self.assign_targets_to_anchors(
            anchors, target_boxes, binary_classes)

        losses = {}
        head_losses, pos_idx, neg_idx = self.head.compute_loss(
            pred_detection, labels, matched_gt_boxes, anchors)
        losses.update(head_losses)

        if self.segmenter is not None:
            losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        if self.patch_classifier is not None:
            losses.update(self.patch_classifier.compute_loss(pred_patch_cls, target_boxes))

        # BI-RADS: RoI-based, uses GT boxes for feature extraction
        if self.birads_classifier is not None:
            losses.update(self.birads_classifier.compute_loss(
                feature_maps_head, target_boxes, target_classes, images.shape[2:]))

        if evaluation:
            prediction = self.postprocess_for_inference(
                images=images,
                pred_detection=pred_detection,
                pred_seg=pred_seg,
                pred_patch_cls=pred_patch_cls,
                feature_maps_head=feature_maps_head,
                anchors=anchors,
            )
        else:
            prediction = None

        return losses, prediction

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Union[List[Tensor], Tensor]]:
        """
        Inference step: detect boxes then classify each with BI-RADS.
        """
        pred_detection, anchors, pred_seg, pred_patch_cls, feature_maps_head = self(images)
        prediction = self.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            pred_seg=pred_seg,
            pred_patch_cls=pred_patch_cls,
            feature_maps_head=feature_maps_head,
            anchors=anchors,
        )
        return prediction

    @torch.no_grad()
    def postprocess_for_inference(
        self,
        images: torch.Tensor,
        pred_detection: Dict[str, torch.Tensor],
        pred_seg: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
        pred_patch_cls: Optional[Dict[str, torch.Tensor]] = None,
        feature_maps_head: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Union[List[Tensor], Tensor]]:
        """
        Postprocess: first decode detections, then classify each box.
        """
        # Parent handles detection + seg + patch_cls
        prediction = super().postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            pred_seg=pred_seg,
            anchors=anchors,
            pred_patch_cls=pred_patch_cls,
        )

        # Classify each predicted box with BI-RADS
        if self.birads_classifier is not None and feature_maps_head is not None:
            pred_boxes = prediction["pred_boxes"]  # List[Tensor [N_i, 6]]
            birads_out = self.birads_classifier(
                feature_maps_head, pred_boxes, images.shape[2:])
            prediction["pred_birads_probs"] = birads_out["birads_probs"]
            prediction["pred_birads_labels"] = birads_out["birads_labels"]

        return prediction

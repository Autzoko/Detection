"""
RetinaUNet V003 - PatchClassifier + BI-RADS Classification Head

Extends V001 (BCE detection + PatchClassifier) with an additional
BI-RADS classification head. Designed for fine-tuning from pretrained
V001 weights.

The detection head remains binary (lesion / no-lesion) with PatchClassifier
for FP suppression. The BI-RADS head is an auxiliary classifier that
predicts BI-RADS class from the same FPN features.

Usage (from scratch):
    nndet_train 100 -o module=RetinaUNetV003 train=v003

Usage (fine-tune from pretrained V001):
    nndet_train 100 -o module=RetinaUNetV003 train=v003 \\
        model_cfg.pretrained_weights=/path/to/v001/model_last.ckpt
"""

import copy
import torch
from typing import Callable, Hashable
from pathlib import Path
from loguru import logger

from nndet.ptmodule.retinaunet.base import RetinaUNetModule

from nndet.core.retina_birads import BiRadsRetinaNet
from nndet.core.boxes.matcher import ATSSMatcher
from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.ops import box_iou
from nndet.core.boxes.anchors import get_anchor_generator, AnchorGeneratorType
from nndet.arch.heads.classifier import BCECLassifier
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.comb import DetectionHeadHNMNative
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.heads.birads_classifier import BiRadsClassifier
from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu, Generator
from nndet.arch.decoder.base import DecoderType

from nndet.ptmodule import MODULE_REGISTRY


@MODULE_REGISTRY.register
class RetinaUNetV003(RetinaUNetModule):
    """
    RetinaUNet with BCE detection + PatchClassifier + BI-RADS classifier.
    Identical to V001 plus an auxiliary BI-RADS classification head.
    """
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = DetectionHeadHNMNative
    head_classifier_cls = BCECLassifier  # Keep binary detection
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg

    def __init__(self, model_cfg: dict, trainer_cfg: dict, plan: dict, **kwargs):
        """
        Override to force binary detection in the evaluator and
        optionally load pretrained V001 weights.
        """
        # Override plan to use 1 class for detection evaluator
        plan = copy.deepcopy(plan)
        plan["architecture"]["classifier_classes"] = 1
        plan["architecture"]["seg_classes"] = 1
        super().__init__(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan, **kwargs)

        # Load pretrained weights if specified
        pretrained_path = model_cfg.get("pretrained_weights", None)
        if pretrained_path is not None:
            pretrained_path = Path(pretrained_path)
            if pretrained_path.exists():
                logger.info(f"V003: Loading pretrained weights from {pretrained_path}")
                ckpt = torch.load(str(pretrained_path), map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                # Load with strict=False to allow missing birads_classifier keys
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"V003: Loaded pretrained weights. "
                    f"Missing keys (new): {len(missing)}, "
                    f"Unexpected keys: {len(unexpected)}"
                )
                if missing:
                    logger.info(f"V003: Missing keys (expected for birads_classifier): "
                                f"{missing}")
            else:
                logger.warning(f"V003: Pretrained weights not found at {pretrained_path}")

    @classmethod
    def _build_birads_classifier(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        decoder: DecoderType,
    ):
        """
        Build BI-RADS classification head (optional).

        Enabled when model_cfg contains 'birads_classifier_kwargs'.

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            decoder: decoder instance

        Returns:
            BiRadsClassifier or None
        """
        kwargs = model_cfg.get("birads_classifier_kwargs", None)
        if kwargs is not None:
            logger.info(f"Building:: birads_classifier {kwargs}")
            birads_classifier = BiRadsClassifier(
                in_channels=decoder.get_channels(),
                decoder_levels=plan_arch["decoder_levels"],
                **kwargs,
            )
        else:
            birads_classifier = None
        return birads_classifier

    @classmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        log_num_anchors: str = None,
        **kwargs,
    ):
        """
        Create RetinaUNet with BI-RADS classifier.

        Overrides the parent to build BiRadsRetinaNet instead of
        BaseRetinaNet.
        """
        logger.info(
            f"Architecture overwrites: {model_cfg['plan_arch_overwrites']} "
            f"Anchor overwrites: {model_cfg['plan_anchors_overwrites']}"
        )
        logger.info(
            f"Building architecture according to plan of "
            f"{plan_arch.get('arch_name', 'not_found')}"
        )
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        plan_anchors.update(model_cfg["plan_anchors_overwrites"])

        # V003: detection head is binary (lesion / no-lesion).
        # The BI-RADS classification is handled by the auxiliary head.
        # Force classifier_classes=1 regardless of what the plan says
        # (plan derives it from dataset.json which now has 3 classes).
        orig_classes = plan_arch.get("classifier_classes", 1)
        plan_arch["classifier_classes"] = 1
        plan_arch["seg_classes"] = 1
        logger.info(
            f"V003: overriding classifier_classes {orig_classes} -> 1 "
            f"(binary detection + auxiliary BI-RADS head)"
        )

        logger.info(
            f"Start channels: {plan_arch['start_channels']}; "
            f"head channels: {plan_arch['head_channels']}; "
            f"fpn channels: {plan_arch['fpn_channels']}"
        )

        _plan_anchors = copy.deepcopy(plan_anchors)
        coder = BoxCoderND(weights=(1.0,) * (plan_arch["dim"] * 2))
        s_param = (
            False
            if ("aspect_ratios" in _plan_anchors)
            and (_plan_anchors["aspect_ratios"] is not None)
            else True
        )
        anchor_generator = get_anchor_generator(
            plan_arch["dim"], s_param=s_param
        )(**_plan_anchors)

        encoder = cls._build_encoder(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        decoder = cls._build_decoder(
            encoder=encoder,
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        matcher = cls.matcher_cls(
            similarity_fn=box_iou,
            **model_cfg["matcher_kwargs"],
        )

        classifier = cls._build_head_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        regressor = cls._build_head_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        head = cls._build_head(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=classifier,
            regressor=regressor,
            coder=coder,
        )
        segmenter = cls._build_segmenter(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            decoder=decoder,
        )
        patch_classifier = cls._build_patch_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            decoder=decoder,
        )
        birads_classifier = cls._build_birads_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            decoder=decoder,
        )

        detections_per_img = plan_arch.get("detections_per_img", 100)
        score_thresh = plan_arch.get("score_thresh", 0)
        topk_candidates = plan_arch.get("topk_candidates", 10000)
        remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
        nms_thresh = plan_arch.get("nms_thresh", 0.6)

        return BiRadsRetinaNet(
            dim=plan_arch["dim"],
            encoder=encoder,
            decoder=decoder,
            head=head,
            anchor_generator=anchor_generator,
            matcher=matcher,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            segmenter=segmenter,
            patch_classifier=patch_classifier,
            birads_classifier=birads_classifier,
            detections_per_img=detections_per_img,
            score_thresh=score_thresh,
            topk_candidates=topk_candidates,
            remove_small_boxes=remove_small_boxes,
            nms_thresh=nms_thresh,
        )

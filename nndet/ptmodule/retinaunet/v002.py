"""
RetinaUNet V002 - BI-RADS Multi-class Detection (from scratch)

Uses CEClassifier (Cross-Entropy / Softmax) instead of BCEClassifier
for native multi-class detection (e.g., BI-RADS 2/3/4 as 3 foreground classes).

No PatchClassifier is used in this variant.

Usage:
    nndet_train 100 -o module=RetinaUNetV002 train=v002
"""

from nndet.ptmodule.retinaunet.base import RetinaUNetModule

from nndet.core.boxes.matcher import ATSSMatcher
from nndet.arch.heads.classifier import CEClassifier
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.comb import DetectionHeadHNMNative
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu

from nndet.ptmodule import MODULE_REGISTRY


@MODULE_REGISTRY.register
class RetinaUNetV002(RetinaUNetModule):
    """
    RetinaUNet with CE (softmax) classifier for multi-class detection.
    Identical to V001 except:
      - head_classifier_cls = CEClassifier (softmax, multi-class)
      - No patch_classifier (train from scratch)
    """
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = DetectionHeadHNMNative
    head_classifier_cls = CEClassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg

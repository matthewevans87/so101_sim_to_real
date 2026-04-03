from abc import ABC, abstractmethod, abstractmethod

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from so101.utils.feature_extraction.spatial_softmax import (
    SpatialSoftmax,
)


class VisionFeatureExtractor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images.

        :param images: A tensor of images to be processed with shape (batch_size, channels, height, width)
        :type images: torch.Tensor
        :return: The tensor of features after processing
        :rtype: torch.Tensor
        """
        pass


class ResNet18SpatialSoftmaxFeatureExtractor(VisionFeatureExtractor):
    def __init__(self, device: str = "cuda"):
        super().__init__()

        self.device = device

        # Pretrained ResNet18 feature extractor (frozen)
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)

        # keep everything up to layer4; drop avgpool and fc
        conv_trunk = nn.Sequential(*list(backbone.children())[:-2])  # (N, 512, Hc, Wc)

        self._vision_backbone = conv_trunk.to(self.device)
        self._vision_backbone.eval()
        for p in self._vision_backbone.parameters():
            p.requires_grad = False

        self._spatial_softmax = SpatialSoftmax().to(self.device)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        # Extract features with frozen ResNet
        with torch.inference_mode():
            conv_feats = self._vision_backbone(images)  # (N, C, Hc, Wc)
            visual_features = self._spatial_softmax(
                conv_feats
            )  # (N, 2C), C=512 → 1024-D

        return visual_features


class CnnSpatialSoftmaxFeatureExtractor(VisionFeatureExtractor):
    """Frozen pretrained CNN feature extractor.

    Drop-in replacement for :class:`ResNet18SpatialSoftmaxFeatureExtractor` that
    loads a :class:`~so101.model.model.MultiTaskCnn`, truncates the projection
    MLP and prediction heads, and feeds the final conv layer output through a
    fresh SpatialSoftmax.  The conv trunk is frozen for the lifetime of this
    object: all parameters are set to ``requires_grad=False`` and the module is
    kept in eval mode.

    Args:
        model: A :class:`~so101.model.model.MultiTaskCnn` instance (weights
            already loaded or to be loaded via :attr:`_backbone`).
        device: PyTorch device string (e.g. ``"cuda:0"``).
    """

    def __init__(self, model: "MultiTaskCnn", device: str = "cuda"):
        from so101.model.model import (
            MultiTaskCnn,
        )  # noqa: F811 (deferred to avoid circular import)

        super().__init__()
        self.device = device

        # Keep a reference to the full backbone for checkpoint loading.
        self._backbone = model.backbone.to(self.device)
        self._backbone.eval()
        for p in self._backbone.parameters():
            p.requires_grad = False

        # Only the conv trunk feeds into a fresh SpatialSoftmax.
        # self._vision_backbone is a live reference to _backbone._conv_trunk, so
        # loading weights into _backbone also updates _vision_backbone.
        self._vision_backbone = self._backbone._conv_trunk
        self._spatial_softmax = SpatialSoftmax().to(self.device)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            conv_feats = self._vision_backbone(images)  # (N, channels[-1], Hc, Wc)
            return self._spatial_softmax(conv_feats)  # (N, 2*channels[-1])

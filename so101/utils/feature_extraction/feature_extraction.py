from abc import ABC, abstractmethod, abstractmethod

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from so101.utils.feature_extraction.spatial_softmax import (
    SpatialSoftmax,
)
from so101.utils.image_processing.image_pipeline import (
    ClampPipelineStep,
    ResizePipelineStep,
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

        # ImageNet normalisation constants — stored as plain tensors so they
        # move to the right device lazily in extract() without being treated
        # as learnable parameters.
        self._imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(1, 3, 1, 1)

        # Preprocessing owned by the extractor: Clamp → Resize to ImageNet resolution.
        self._clamp = ClampPipelineStep(0.0, 1.0)
        self._resize = ResizePipelineStep((224, 224))

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        # Clamp and resize before normalisation.  images is (N, 3, H, W) float
        # produced by the shared pipeline (Uint8ToFloatCHW → DR augmentations).
        images = self._clamp.process(images)
        images = self._resize.process(images)

        # Apply ImageNet normalisation.
        mean = self._imagenet_mean.to(images.device)
        std = self._imagenet_std.to(images.device)
        images = (images - mean) / std

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

    def __init__(self, model: "MultiTaskCnn", device: str = "cuda", target_size: tuple[int, int] = (108, 192)):
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

        # Preprocessing owned by the extractor: Clamp → Resize to policy resolution.
        self._clamp = ClampPipelineStep(0.0, 1.0)
        self._resize = ResizePipelineStep(target_size)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        images = self._clamp.process(images)
        images = self._resize.process(images)
        with torch.inference_mode():
            conv_feats = self._vision_backbone(images)  # (N, channels[-1], Hc, Wc)
            return self._spatial_softmax(conv_feats)  # (N, 2*channels[-1])



from abc import ABC, abstractmethod, abstractmethod

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from so101_rl.nnmodules.spatial_softmax import (
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
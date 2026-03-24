from abc import ABC, abstractmethod, abstractmethod

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from so101_utils.feature_extraction.spatial_softmax import (
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


class TrainableCnnFeatureExtractor(nn.Module):
    """Trainable CNN backbone: configurable conv layers with stride-based downsampling,
    followed by SpatialSoftmax and a small MLP projection.

    This module is intended to be embedded inside a skrl policy model so that
    backpropagation from the PPO loss updates CNN weights during training.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        channels: Output channels for each conv layer.
        kernel_sizes: Kernel size for each conv layer.
        strides: Stride for each conv layer (stride-based downsampling, no pooling).
        mlp_hidden_dims: Hidden layer sizes for the MLP projection after SpatialSoftmax.
        output_dim: Final feature dimensionality.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list | None = None,
        kernel_sizes: list | None = None,
        strides: list | None = None,
        mlp_hidden_dims: list | None = None,
        output_dim: int = 256,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 128]
        if kernel_sizes is None:
            kernel_sizes = [8, 4, 3, 3]
        if strides is None:
            strides = [4, 2, 2, 1]
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256]

        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                f"channels, kernel_sizes, and strides must have equal length; "
                f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}."
            )

        self._output_dim = output_dim

        # Conv trunk: Conv2d → BatchNorm2d → ReLU blocks
        conv_layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=k,
                        stride=s,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self._conv_trunk = nn.Sequential(*conv_layers)

        self._spatial_softmax = SpatialSoftmax()

        # MLP projection: SpatialSoftmax outputs 2 * channels[-1] features
        spatial_dim = 2 * channels[-1]
        mlp_layers: list = []
        prev = spatial_dim
        for h in mlp_hidden_dims:
            mlp_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        mlp_layers.append(nn.Linear(prev, output_dim))
        self._mlp = nn.Sequential(*mlp_layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: (N, C, H, W) float tensor, values in [0, 1]
        :return: (N, output_dim) feature tensor; gradients flow freely for PPO updates
        """
        conv_feats = self._conv_trunk(images)  # (N, channels[-1], Hc, Wc)
        spatial_feats = self._spatial_softmax(conv_feats)  # (N, 2*channels[-1])
        return self._mlp(spatial_feats)  # (N, output_dim)

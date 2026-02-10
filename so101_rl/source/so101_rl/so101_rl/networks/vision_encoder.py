"""Vision encoder with spatial soft-argmax and task-specific MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SpatialSoftArgmax(nn.Module):
    """Spatial soft-argmax layer for extracting 2D keypoints from feature maps.

    Converts spatial feature maps into a set of 2D coordinates (keypoints) by
    computing the spatial expectation of each feature channel.

    Reference: "End-to-End Training of Deep Visuomotor Policies" (Levine et al.)
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, normalize coordinates to [-1, 1] range
        """
        super().__init__()
        self.normalize = normalize

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Extract keypoints from feature maps.

        Args:
            features: (N, C, H, W) feature maps

        Returns:
            keypoints: (N, C, 2) - (x, y) coordinates for each of C channels
        """
        N, C, H, W = features.shape

        # Create coordinate grids
        device = features.device
        dtype = features.dtype

        # y coordinates: (H, W) - row indices
        # x coordinates: (H, W) - column indices
        pos_y, pos_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )

        # Normalize to [-1, 1] if requested
        if self.normalize:
            pos_x = (pos_x / (W - 1)) * 2 - 1  # (H, W)
            pos_y = (pos_y / (H - 1)) * 2 - 1  # (H, W)

        # Flatten spatial dimensions: (N, C, H*W)
        features_flat = features.view(N, C, H * W)

        # Apply softmax to get spatial attention weights: (N, C, H*W)
        attention = F.softmax(features_flat, dim=2)

        # Flatten position grids: (H*W,)
        pos_x_flat = pos_x.reshape(-1)  # (H*W,)
        pos_y_flat = pos_y.reshape(-1)  # (H*W,)

        # Compute expected x coordinate: sum over spatial locations
        # (N, C, H*W) * (H*W,) -> (N, C)
        expected_x = torch.sum(attention * pos_x_flat, dim=2)
        expected_y = torch.sum(attention * pos_y_flat, dim=2)

        # Stack: (N, C, 2)
        keypoints = torch.stack([expected_x, expected_y], dim=2)

        return keypoints


class VisionEncoder(nn.Module):
    """Vision encoder with ResNet backbone, spatial soft-argmax, and task MLP.

    Architecture:
        1. Frozen pretrained ResNet18 → spatial feature maps (N, 512, H/32, W/32)
        2. [Optional] Spatial soft-argmax → keypoints (N, num_keypoints, 2)
        3. Task-specific MLP → learned features (N, output_dim)

    The MLP learns to extract task-relevant information (e.g., cube distance,
    location) from visual features, improving sample efficiency vs raw CNN features.
    """

    def __init__(
        self,
        output_dim: int = 64,
        num_keypoints: int = 16,
        mlp_hidden_dims: list[int] | None = None,
        use_spatial_softargmax: bool = True,
        freeze_backbone: bool = True,
        use_depthwise_separable: bool = False,  # P2: Use depthwise-separable conv (fewer params)
        use_mlp_skip_connections: bool = False,  # P2: Add skip connections in MLP
        device: str = "cuda",
    ):
        """
        Args:
            output_dim: Dimension of output task features
            num_keypoints: Number of keypoints to extract (if using soft-argmax)
            mlp_hidden_dims: Hidden layer sizes for task MLP
            use_spatial_softargmax: If True, use keypoints; if False, use raw pooled features
            freeze_backbone: If True, freeze ResNet weights (recommended)
            use_depthwise_separable: If True, use depthwise-separable conv for channel reduction
            use_mlp_skip_connections: If True, add skip connections in MLP layers
            device: Device to create the model on
        """
        super().__init__()

        self.output_dim = output_dim
        self.num_keypoints = num_keypoints
        self.use_spatial_softargmax = use_spatial_softargmax
        self.use_depthwise_separable = use_depthwise_separable
        self.use_mlp_skip_connections = use_mlp_skip_connections
        self.device = device

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128]
        self.mlp_hidden_dims = mlp_hidden_dims

        # 1. ResNet18 backbone
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)

        # Remove final pooling and classifier to get spatial feature maps
        # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # We want features after layer4, before avgpool
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()  # Note: no dropout/BN in feature extraction, .eval() mainly for consistency

        # ResNet18 layer4 outputs 512 channels
        self.feature_channels = 512

        # 2. Feature extraction path
        if use_spatial_softargmax:
            # Reduce feature channels to num_keypoints before soft-argmax
            # This learns which features to track as keypoints

            if use_depthwise_separable:
                # P2 Optimization: Depthwise-separable convolution (fewer parameters)
                # Standard conv: 512 × num_keypoints params
                # Depthwise-sep: 512 + 512 × num_keypoints params (~50% reduction for small num_keypoints)
                self.channel_reducer = nn.Sequential(
                    # Depthwise: each input channel separately
                    nn.Conv2d(
                        self.feature_channels,
                        self.feature_channels,
                        kernel_size=1,
                        groups=self.feature_channels,
                        bias=False,
                    ),
                    # Pointwise: mix channels
                    nn.Conv2d(
                        self.feature_channels, num_keypoints, kernel_size=1, bias=True
                    ),
                )
            else:
                # Standard 1×1 convolution
                # Conv with bias for better expressiveness (learns per-keypoint offset)
                self.channel_reducer = nn.Conv2d(
                    self.feature_channels, num_keypoints, kernel_size=1, bias=True
                )

            self.soft_argmax = SpatialSoftArgmax(normalize=True)

            # Input to MLP: num_keypoints * 2 (x, y for each keypoint)
            mlp_input_dim = num_keypoints * 2
        else:
            # Use global average pooling + raw features
            self.channel_reducer = None
            self.soft_argmax = None
            mlp_input_dim = self.feature_channels

        # 3. Task-specific MLP
        mlp_layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_dim = hidden_dim

        # Final output layer
        mlp_layers.append(nn.Linear(prev_dim, output_dim))

        self.task_mlp = nn.Sequential(*mlp_layers)

        # Auxiliary prediction heads (optional): learn specific features
        # Predicted distance scalar, cube position (x,y,z), and EE height scalar
        self.aux_dist_head = nn.Linear(output_dim, 1)
        self.aux_pos_head = nn.Linear(output_dim, 3)
        self.aux_eez_head = nn.Linear(output_dim, 1)

        # Move to device
        self.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract task-specific visual features from images.

        Args:
            images: (N, 3, H, W) normalized RGB images

        Returns:
            features: (N, output_dim) task-specific learned features
        """
        # 1. ResNet backbone: (N, 3, H, W) -> (N, 512, H/32, W/32)
        with torch.set_grad_enabled(self.training and self.backbone.training):
            spatial_features = self.backbone(images)

        # 2. Extract features
        if self.use_spatial_softargmax:
            # Reduce channels: (N, 512, H', W') -> (N, num_keypoints, H', W')
            reduced_features = self.channel_reducer(spatial_features)

            # Soft-argmax: (N, num_keypoints, H', W') -> (N, num_keypoints, 2)
            keypoints = self.soft_argmax(reduced_features)

            # Flatten: (N, num_keypoints, 2) -> (N, num_keypoints * 2)
            mlp_input = keypoints.reshape(keypoints.shape[0], -1)
        else:
            # Global average pooling: (N, 512, H', W') -> (N, 512)
            mlp_input = F.adaptive_avg_pool2d(spatial_features, (1, 1))
            mlp_input = mlp_input.view(mlp_input.shape[0], -1)

        # 3. Task MLP
        task_features = self.task_mlp(mlp_input)

        return task_features

    def predict_aux(self, task_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict auxiliary targets from task features.

        Returns a dict with keys: 'dist', 'cube_pos', 'ee_height'.
        """
        return {
            "dist": self.aux_dist_head(task_features),
            "cube_pos": self.aux_pos_head(task_features),
            "ee_height": self.aux_eez_head(task_features),
        }

    def aux_loss(
        self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute MSE loss for auxiliary predictions against targets.

        This is intended for use in reward shaping (not actor observations) or
        for external training pipelines."""
        loss = 0.0
        if "dist" in preds and "dist" in targets:
            loss = loss + torch.mean((preds["dist"] - targets["dist"]) ** 2)
        if "cube_pos" in preds and "cube_pos" in targets:
            loss = loss + torch.mean((preds["cube_pos"] - targets["cube_pos"]) ** 2)
        if "ee_height" in preds and "ee_height" in targets:
            loss = loss + torch.mean((preds["ee_height"] - targets["ee_height"]) ** 2)
        return loss

    def get_keypoints(self, images: torch.Tensor) -> torch.Tensor | None:
        """Helper to extract keypoints for visualization.

        Args:
            images: (N, 3, H, W) normalized RGB images

        Returns:
            keypoints: (N, num_keypoints, 2) or None if not using soft-argmax
        """
        if not self.use_spatial_softargmax:
            return None

        with torch.no_grad():
            spatial_features = self.backbone(images)
            reduced_features = self.channel_reducer(spatial_features)
            keypoints = self.soft_argmax(reduced_features)

        return keypoints

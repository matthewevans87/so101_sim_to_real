import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SpatialSoftmax(nn.Module):
    """Spatial softmax layer that converts (N, C, H, W) to (N, 2C) by computing
    expected spatial coordinates for each channel."""

    def __init__(self):
        super().__init__()
        # buffers will be initialized on first forward
        self.register_buffer("pos_x", None, persistent=False)
        self.register_buffer("pos_y", None, persistent=False)

    def _build_grid(self, H: int, W: int, device):
        # [-1, 1] coordinate grid
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij",
        )
        self.pos_x = xs.reshape(1, 1, H * W)
        self.pos_y = ys.reshape(1, 1, H * W)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (N, C, H, W)
        N, C, H, W = features.shape

        if self.pos_x is None or self.pos_x.shape[-1] != H * W:
            self._build_grid(H, W, features.device)

        feats = features.view(N, C, H * W)  # (N, C, HW)
        attn = torch.softmax(feats, dim=-1)  # attention over spatial locations

        # expected coordinates per channel
        exp_x = torch.sum(attn * self.pos_x, dim=-1)  # (N, C)
        exp_y = torch.sum(attn * self.pos_y, dim=-1)  # (N, C)

        # Output: (N, 2C), each channel -> (x, y)
        return torch.cat([exp_x, exp_y], dim=-1)


class VisionFcPolicy(nn.Module):
    """
    Vision-based policy using frozen ResNet18 with FC layers (512-D features).
    Architecture: Camera → ResNet18(frozen, with avgpool) → 512-D → concat with joint positions → MLP
    """

    def __init__(
        self,
        num_joints: int,
        act_dim: int,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.num_joints = num_joints

        # Frozen ResNet18 backbone (with avgpool, outputs 512-D)
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()  # Remove final classifier
        self._vision_backbone = backbone.to(device)
        self._vision_backbone.eval()
        for p in self._vision_backbone.parameters():
            p.requires_grad = False

        # ImageNet normalization (cached)
        self.register_buffer(
            "_img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Policy network: [512 vision + num_joints] → hidden1 → hidden2 → hidden3 → actions
        self.fc1 = nn.Linear(512 + num_joints, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, act_dim)

    def _preprocess_camera(self, camera_rgb: torch.Tensor) -> torch.Tensor:
        """Preprocess camera image for ResNet18."""
        # Handle single image case
        if camera_rgb.dim() == 3:
            camera_rgb = camera_rgb.unsqueeze(0)

        # Convert to float and normalize to [0, 1] if needed
        if camera_rgb.dtype == torch.uint8:
            images = camera_rgb.float() / 255.0
        else:
            images = camera_rgb

        # Convert from (N, H, W, 3) to (N, 3, H, W)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        # Resize to 224x224 for ResNet18
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )

        # ImageNet normalization
        images = (images - self._img_mean) / self._img_std
        return images

    @torch.no_grad()
    def forward(
        self, camera_rgb: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for FC-based vision policy.

        Args:
            camera_rgb: Raw camera RGB, shape (N, H, W, 3) or (H, W, 3)
            joint_positions: Joint positions, shape (N, num_joints) or (num_joints,)

        Returns:
            Actions in [-1, 1], shape (N, act_dim) or (act_dim,)
        """
        single_obs = joint_positions.dim() == 1
        if single_obs:
            joint_positions = joint_positions.unsqueeze(0)

        # Extract 512-D pooled features
        images = self._preprocess_camera(camera_rgb)
        with torch.inference_mode():
            visual_features = self._vision_backbone(images)  # (N, 512)

        # Concatenate and pass through policy network
        obs = torch.cat([visual_features, joint_positions], dim=-1)
        x = F.elu(self.fc1(obs))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        actions = torch.tanh(x)

        if single_obs:
            actions = actions.squeeze(0)
        return actions


class VisionConvPolicy(nn.Module):
    """
    Vision-based policy using frozen ResNet18 conv features + spatial softmax (1024-D features).
    Architecture: Camera → ResNet18 conv(frozen) → Spatial Softmax → 1024-D → concat with joint positions → MLP
    """

    def __init__(
        self,
        num_joints: int,
        act_dim: int,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.num_joints = num_joints

        # Frozen ResNet18 conv backbone (no avgpool/fc, outputs 512 x H x W)
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        conv_trunk = nn.Sequential(*list(backbone.children())[:-2])
        self._vision_backbone = conv_trunk.to(device)
        self._vision_backbone.eval()
        for p in self._vision_backbone.parameters():
            p.requires_grad = False

        # Spatial softmax: (N, 512, H, W) → (N, 1024)
        self._spatial_softmax = SpatialSoftmax()

        # ImageNet normalization (cached)
        self.register_buffer(
            "_img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Policy network: [1024 vision + num_joints] → hidden1 → hidden2 → hidden3 → actions
        self.fc1 = nn.Linear(1024 + num_joints, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, act_dim)

    def _preprocess_camera(self, camera_rgb: torch.Tensor) -> torch.Tensor:
        """Preprocess camera image for ResNet18."""
        # Handle single image case
        if camera_rgb.dim() == 3:
            camera_rgb = camera_rgb.unsqueeze(0)

        # Convert to float and normalize to [0, 1] if needed
        if camera_rgb.dtype == torch.uint8:
            images = camera_rgb.float() / 255.0
        else:
            images = camera_rgb

        # Convert from (N, H, W, 3) to (N, 3, H, W)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        # Resize to 224x224 for ResNet18
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )

        # ImageNet normalization
        images = (images - self._img_mean) / self._img_std
        return images

    @torch.no_grad()
    def forward(
        self, camera_rgb: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for conv-based vision policy.

        Args:
            camera_rgb: Raw camera RGB, shape (N, H, W, 3) or (H, W, 3)
            joint_positions: Joint positions, shape (N, num_joints) or (num_joints,)

        Returns:
            Actions in [-1, 1], shape (N, act_dim) or (act_dim,)
        """
        single_obs = joint_positions.dim() == 1
        if single_obs:
            joint_positions = joint_positions.unsqueeze(0)

        # Extract 1024-D spatial features
        images = self._preprocess_camera(camera_rgb)
        with torch.inference_mode():
            conv_feats = self._vision_backbone(images)  # (N, 512, Hc, Wc)
            visual_features = self._spatial_softmax(conv_feats)  # (N, 1024)

        # Concatenate and pass through policy network
        obs = torch.cat([visual_features, joint_positions], dim=-1)
        x = F.elu(self.fc1(obs))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        actions = torch.tanh(x)

        if single_obs:
            actions = actions.squeeze(0)
        return actions

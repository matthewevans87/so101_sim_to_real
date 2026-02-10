import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder


class MultiViewVisionEncoder(nn.Module):
    """Two-view encoder: gripper-mounted and overhead cameras.

    Each view is encoded via VisionEncoder to a feature vector; features are fused
    via a small MLP to produce final task features of fixed dimension.
    """

    def __init__(
        self,
        output_dim: int = 64,
        per_view_output_dim: int = 64,
        num_keypoints: int = 16,
        mlp_hidden_dims: list[int] | None = None,
        use_spatial_softargmax: bool = True,
        freeze_backbone: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128]

        # Per-view encoders
        self.gripper_encoder = VisionEncoder(
            output_dim=per_view_output_dim,
            num_keypoints=num_keypoints,
            mlp_hidden_dims=mlp_hidden_dims,
            use_spatial_softargmax=use_spatial_softargmax,
            freeze_backbone=freeze_backbone,
            device=device,
        )
        self.overhead_encoder = VisionEncoder(
            output_dim=per_view_output_dim,
            num_keypoints=num_keypoints,
            mlp_hidden_dims=mlp_hidden_dims,
            use_spatial_softargmax=use_spatial_softargmax,
            freeze_backbone=freeze_backbone,
            device=device,
        )

        # Fusion MLP
        fused_in = per_view_output_dim * 2
        fusion_layers = [
            nn.Linear(fused_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        ]
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        self.to(device)

    def forward(self, gripper_images: torch.Tensor, overhead_images: torch.Tensor) -> torch.Tensor:
        g_feat = self.gripper_encoder(gripper_images)
        o_feat = self.overhead_encoder(overhead_images)
        fused = torch.cat([g_feat, o_feat], dim=-1)
        return self.fusion_mlp(fused)

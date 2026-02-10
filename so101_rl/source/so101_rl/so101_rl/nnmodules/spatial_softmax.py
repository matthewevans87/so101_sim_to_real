import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
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

import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pos_x", None, persistent=False)
        self.register_buffer("pos_y", None, persistent=False)

    def _build_grid(self, height: int, width: int, device: torch.device) -> None:
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device),
            indexing="ij",
        )
        self.pos_x = xs.reshape(1, 1, height * width)
        self.pos_y = ys.reshape(1, 1, height * width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = features.shape

        if self.pos_x is None or self.pos_x.shape[-1] != height * width:
            self._build_grid(height, width, features.device)

        flattened = features.view(batch_size, channels, height * width)
        attention = torch.softmax(flattened, dim=-1)

        expected_x = torch.sum(attention * self.pos_x, dim=-1)
        expected_y = torch.sum(attention * self.pos_y, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)

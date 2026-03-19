"""Visualization helpers for CNN feature maps and SpatialSoftmax keypoints.

Both functions accept CPU float tensors and return ``uint8 (3, H, W)`` tensors
ready for direct use with ``SummaryWriter.add_image``.

Coordinate convention (matches :class:`SpatialSoftmax`):
  x in [-1, 1] → horizontal (left=-1, right=+1)
  y in [-1, 1] → vertical   (top=-1,  bottom=+1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _build_jet_lut() -> torch.Tensor:
    """Return a ``(256, 3)`` float32 tensor of the jet colourmap in ``[0, 1]``."""
    t = torch.linspace(0.0, 1.0, 256)
    r = torch.clamp(1.5 - torch.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = torch.clamp(1.5 - torch.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = torch.clamp(1.5 - torch.abs(4.0 * t - 1.0), 0.0, 1.0)
    return torch.stack([r, g, b], dim=1)  # (256, 3)


_JET_LUT: torch.Tensor = _build_jet_lut()


def draw_activation_heatmap_overlay(
    image_chw: torch.Tensor,
    conv_feats_chw: torch.Tensor,
    alpha: float = 0.55,
) -> torch.Tensor:
    """Alpha-blend a jet activation heatmap over a camera image.

    The heatmap is the mean activation across all channels of the final conv
    trunk output, upsampled to the input image resolution.

    Args:
        image_chw: ``(3, H, W)`` float tensor in ``[0, 1]``.  Must be on CPU.
        conv_feats_chw: ``(C, Hc, Wc)`` float tensor (raw conv trunk output
            before SpatialSoftmax).  Must be on CPU.
        alpha: Weight of the heatmap in the blend.  ``0`` = image only,
            ``1`` = heatmap only.

    Returns:
        ``uint8 (3, H, W)`` tensor ready for ``SummaryWriter.add_image``.
    """
    _, H, W = image_chw.shape

    # Mean over channels → (Hc, Wc); retain activation polarity info.
    heatmap = conv_feats_chw.float().mean(0)  # (Hc, Wc)

    # Normalise to [0, 1]; guard against constant activation maps.
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = torch.zeros_like(heatmap)

    # Bilinear upsample to input image resolution.
    heatmap_up = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),  # (1, 1, Hc, Wc)
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # (H, W)

    # Map scalar intensities through the jet LUT.
    idx = (heatmap_up * 255).long().clamp(0, 255)  # (H, W) int indices
    jet_rgb = _JET_LUT[idx].permute(2, 0, 1)  # (3, H, W) float [0, 1]

    # Alpha-blend with the original image.
    blend = alpha * jet_rgb + (1.0 - alpha) * image_chw.float().clamp(0.0, 1.0)
    return (blend * 255).clamp(0, 255).byte()  # uint8 (3, H, W)


def draw_keypoints_overlay(
    image_chw: torch.Tensor,
    keypoints_2c: torch.Tensor,
    radius: int = 3,
) -> torch.Tensor:
    """Draw SpatialSoftmax keypoints as coloured squares over a camera image.

    ``keypoints_2c`` follows the layout produced by :class:`SpatialSoftmax``:
    ``[x_0, …, x_{C-1}, y_0, …, y_{C-1}]`` where each coordinate is in
    ``[-1, 1]`` (x = horizontal left→right, y = vertical top→bottom).

    Args:
        image_chw: ``(3, H, W)`` float tensor in ``[0, 1]``.  Must be on CPU.
        keypoints_2c: ``(2C,)`` float tensor of normalised keypoint coordinates.
            Must be on CPU.
        radius: Half-size (pixels) of each drawn square.  Default 3 → 7×7 px.

    Returns:
        ``uint8 (3, H, W)`` tensor ready for ``SummaryWriter.add_image``.
    """
    _, H, W = image_chw.shape
    C = keypoints_2c.shape[0] // 2

    xs = keypoints_2c[:C]  # (C,) horizontal coordinates in [-1, 1]
    ys = keypoints_2c[C:]  # (C,) vertical   coordinates in [-1, 1]

    # Map normalised [-1, 1] → pixel coordinates.
    # x → column (0 = left),  y → row (0 = top)
    px = ((xs + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)  # (C,)
    py = ((ys + 1.0) * 0.5 * (H - 1)).long().clamp(0, H - 1)  # (C,)

    img_u8 = (image_chw.float().clamp(0.0, 1.0) * 255).byte().clone()  # (3, H, W)
    colours = _keypoint_colours(C)  # (C, 3) uint8

    r = max(1, radius)
    for i in range(C):
        x_c, y_c = int(px[i].item()), int(py[i].item())
        x0 = max(0, x_c - r)
        x1 = min(W, x_c + r + 1)
        y0 = max(0, y_c - r)
        y1 = min(H, y_c + r + 1)
        img_u8[:, y0:y1, x0:x1] = colours[i].view(3, 1, 1)

    return img_u8


def _keypoint_colours(n: int) -> torch.Tensor:
    """Return an ``(n, 3)`` uint8 tensor of evenly-spaced HSV hues converted to RGB.

    Uses HSV with full saturation and value (s=1, v=1) so every keypoint gets
    a distinct, highly saturated colour.
    """
    if n == 0:
        return torch.zeros(0, 3, dtype=torch.uint8)

    # Evenly space hues across [0, 1) to avoid wrapping back to the same colour.
    hues = torch.linspace(0.0, 1.0 - 1.0 / n, n)  # (n,)
    h6 = hues * 6.0
    i = h6.long()
    f = h6 - i.float()

    # Standard HSV→RGB for s=1, v=1:
    #   sector i → (r, g, b) depend on f only
    p = torch.zeros_like(f)  # 0
    q = 1.0 - f  # 1 - f
    t = f  # f
    one = torch.ones_like(f)  # 1

    i6 = i % 6
    r = torch.where(
        i6 == 0,
        one,
        torch.where(
            i6 == 1,
            q,
            torch.where(
                i6 == 2, p, torch.where(i6 == 3, p, torch.where(i6 == 4, t, one))
            ),
        ),
    )
    g = torch.where(
        i6 == 0,
        t,
        torch.where(
            i6 == 1,
            one,
            torch.where(
                i6 == 2, one, torch.where(i6 == 3, q, torch.where(i6 == 4, p, p))
            ),
        ),
    )
    b = torch.where(
        i6 == 0,
        p,
        torch.where(
            i6 == 1,
            p,
            torch.where(
                i6 == 2, t, torch.where(i6 == 3, one, torch.where(i6 == 4, one, q))
            ),
        ),
    )

    rgb = torch.stack([r, g, b], dim=1)  # (n, 3) float [0, 1]
    return (rgb * 255).clamp(0, 255).byte()

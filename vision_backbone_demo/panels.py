"""Panel rendering functions.

Each function takes raw tensors and returns an ``(H, W, 3)`` uint8 NumPy array
suitable for display and compositing.  All operations are done on single frames
(batch dimension = 1).

The algorithms are ported from
``so101_rl/source/so101_rl/so101_rl/viz/vision_debug.py``
(``VisionDebugLogger._write_*`` methods) with TensorBoard calls replaced by
NumPy array returns.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Raw frame
# ---------------------------------------------------------------------------


def render_raw(rgb_uint8: np.ndarray) -> np.ndarray:
    """Return the raw HWC uint8 RGB frame unchanged.

    Args:
        rgb_uint8: ``(H, W, 3)`` uint8 NumPy array.

    Returns:
        ``(H, W, 3)`` uint8 array.
    """
    return rgb_uint8.copy()


# ---------------------------------------------------------------------------
# Pipelined frame
# ---------------------------------------------------------------------------


def render_pipelined(pipe_chw_float: torch.Tensor) -> np.ndarray:
    """Normalise the pipelined CHW float tensor to [0, 1] and return HWC uint8.

    The pipeline may have applied ImageNet normalisation, so we do a
    per-image min-max normalisation for display only (same as
    ``VisionDebugLogger._write_pipelined_image``).

    Args:
        pipe_chw_float: ``(3, H, W)`` float tensor on any device.

    Returns:
        ``(H, W, 3)`` uint8 array.
    """
    img = pipe_chw_float.detach().cpu().float()  # (3, H, W)
    lo = img.flatten().min()
    hi = img.flatten().max()
    img = (img - lo) / (hi - lo + 1e-6)
    img = img.clamp(0.0, 1.0)
    # CHW → HWC
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Conv-layer feature-map grid
# ---------------------------------------------------------------------------


def render_conv_grid(
    feature_map: torch.Tensor,
    max_channels: int,
    tile_size: int,
) -> np.ndarray:
    """Tile the first ``max_channels`` channels of a feature map into a square grid.

    Each channel is normalised independently to [0, 1] and rendered as a
    greyscale tile.  Tiles are arranged in a grid with ``ceil(sqrt(N))`` columns.

    Args:
        feature_map: ``(1, C, Hc, Wc)`` float tensor (batch size = 1).
        max_channels: Maximum number of channels to render.
        tile_size: Each channel tile is resized to ``(tile_size, tile_size)`` px.

    Returns:
        ``(grid_H, grid_W, 3)`` uint8 array.
    """
    fmap = feature_map[0].detach().cpu().float()  # (C, Hc, Wc)
    C = fmap.shape[0]
    num_ch = min(C, max_channels)
    fmap = fmap[:num_ch]  # (num_ch, Hc, Wc)

    # Per-channel normalisation to [0, 1]
    flat = fmap.reshape(num_ch, -1)
    lo = flat.min(dim=1).values[:, None, None]
    hi = flat.max(dim=1).values[:, None, None]
    fmap = (fmap - lo) / (hi - lo + 1e-6)  # (num_ch, Hc, Wc)

    # Resize each channel tile to tile_size × tile_size
    fmap_4d = fmap.unsqueeze(1)  # (num_ch, 1, Hc, Wc)
    fmap_4d = F.interpolate(
        fmap_4d, size=(tile_size, tile_size), mode="bilinear", align_corners=False
    )  # (num_ch, 1, tile_size, tile_size)

    # Arrange into a square grid
    ncols = max(1, math.ceil(math.sqrt(num_ch)))
    nrows = max(1, math.ceil(num_ch / ncols))

    # Pad to fill the grid
    pad = nrows * ncols - num_ch
    if pad > 0:
        fmap_4d = torch.cat([fmap_4d, torch.zeros(pad, 1, tile_size, tile_size)], dim=0)

    tiles = fmap_4d.squeeze(1)  # (nrows*ncols, tile_size, tile_size)
    rows = tiles.reshape(nrows, ncols, tile_size, tile_size)
    # (nrows, tile_size, ncols, tile_size) → (nrows*tile_size, ncols*tile_size)
    grid = rows.permute(0, 2, 1, 3).reshape(nrows * tile_size, ncols * tile_size)

    # Greyscale → RGB
    grid_np = (grid.numpy() * 255.0).astype(np.uint8)
    return np.stack([grid_np, grid_np, grid_np], axis=-1)


# ---------------------------------------------------------------------------
# Activation heatmap
# ---------------------------------------------------------------------------


def render_heatmap(
    pipe_chw_float: torch.Tensor,
    last_feature_map: torch.Tensor,
) -> np.ndarray:
    """Alpha-blend a red activation heatmap over the pipelined image.

    Ports ``VisionDebugLogger._write_activation_heatmap``.

    Args:
        pipe_chw_float: ``(3, H, W)`` float pipelined image tensor.
        last_feature_map: ``(1, C, Hc, Wc)`` float final conv feature map.

    Returns:
        ``(H, W, 3)`` uint8 blended image.
    """
    last_fmap = last_feature_map[0:1].detach().cpu().float()  # (1, C, Hc, Wc)
    H, W = pipe_chw_float.shape[1], pipe_chw_float.shape[2]

    # Mean over channels → (1, 1, Hc, Wc)
    heat = last_fmap.mean(dim=1, keepdim=True)

    # Upsample to pipelined resolution
    heat = F.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)

    # Normalise to [0, 1]
    flat = heat.reshape(1, -1)
    lo = flat.min(dim=1).values[:, None, None, None]
    hi = flat.max(dim=1).values[:, None, None, None]
    heat = (heat - lo) / (hi - lo + 1e-6)  # (1, 1, H, W)

    # Red overlay
    red_overlay = torch.cat(
        [heat, torch.zeros_like(heat), torch.zeros_like(heat)], dim=1
    )  # (1, 3, H, W)

    # Normalise bg to [0, 1]
    bg = pipe_chw_float.detach().cpu().float().unsqueeze(0)  # (1, 3, H, W)
    bg_flat = bg.flatten(1)
    lo_bg = bg_flat.min(dim=1).values[:, None, None, None]
    hi_bg = bg_flat.max(dim=1).values[:, None, None, None]
    bg = (bg - lo_bg) / (hi_bg - lo_bg + 1e-6)

    blended = (0.5 * bg + 0.5 * red_overlay).clamp(0.0, 1.0)  # (1, 3, H, W)

    out = blended[0].permute(1, 2, 0).numpy()  # (H, W, 3)
    return (out * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# SpatialSoftmax keypoints
# ---------------------------------------------------------------------------


def render_keypoints(
    pipe_chw_float: torch.Tensor,
    last_feature_map: torch.Tensor,
) -> np.ndarray:
    """Draw SpatialSoftmax keypoints over the pipelined image.

    Ports ``VisionDebugLogger._write_keypoints``.

    Each channel of the final conv feature map contributes one (x, y) keypoint
    via SpatialSoftmax.  Keypoints are drawn as filled white squares (radius 2 px).

    Args:
        pipe_chw_float: ``(3, H, W)`` float pipelined image tensor.
        last_feature_map: ``(1, C, Hc, Wc)`` float final conv feature map.

    Returns:
        ``(H, W, 3)`` uint8 image with keypoints overlaid.
    """
    from so101.utils.feature_extraction.spatial_softmax import SpatialSoftmax

    last_fmap = last_feature_map[0:1].detach()  # (1, C, Hc, Wc); keep on GPU

    ss = SpatialSoftmax().to(last_fmap.device)
    with torch.no_grad():
        coords = ss(last_fmap)  # (1, 2*C)

    C = last_fmap.shape[1]
    kp_x = coords[0, :C].cpu().float()  # (C,) in [-1, 1]
    kp_y = coords[0, C:].cpu().float()  # (C,) in [-1, 1]

    H, W = pipe_chw_float.shape[1], pipe_chw_float.shape[2]

    # Un-normalise to pixel space
    px = ((kp_x + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)  # (C,)
    py = ((kp_y + 1.0) * 0.5 * (H - 1)).long().clamp(0, H - 1)  # (C,)

    # Build normalised background canvas
    bg = pipe_chw_float.detach().cpu().float().unsqueeze(0)  # (1, 3, H, W)
    bg_flat = bg.flatten(1)
    lo = bg_flat.min(dim=1).values[:, None, None, None]
    hi = bg_flat.max(dim=1).values[:, None, None, None]
    canvas = (bg - lo) / (hi - lo + 1e-6)  # (1, 3, H, W)
    canvas = canvas.clamp(0.0, 1.0)

    r = 2
    for kp_i in range(C):
        cx = int(px[kp_i].item())
        cy = int(py[kp_i].item())
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, W)
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, H)
        canvas[0, :, y0:y1, x0:x1] = 1.0  # white

    out = canvas[0].permute(1, 2, 0).numpy()  # (H, W, 3)
    return (out * 255.0).astype(np.uint8)

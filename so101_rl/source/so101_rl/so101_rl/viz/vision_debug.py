"""VisionDebugLogger — TensorBoard visualisations of CNN vision features.

Visualisations logged (each individually gated by the YAML config):

``VisionDebug/raw_image``
    The uint8 RGB frame straight out of the TiledCamera, before any processing.

``VisionDebug/pipelined_image``
    The float CHW frame after the full image pipeline (resized, normalised, …).

``VisionDebug/conv_layer_maps/<layer_idx>``
    A tile grid of the first *max_channels* feature maps from each conv block's
    ReLU output.  Works only for :class:`CnnSpatialSoftmaxFeatureExtractor`
    (silently skipped for ResNet18).

``VisionDebug/activation_heatmap``
    Mean activation over channels of the final conv block, upsampled to the
    pipelined image resolution and alpha-blended over it as a red overlay.

``VisionDebug/keypoints``
    SpatialSoftmax keypoints (one dot per channel of the final conv block)
    drawn over the pipelined image as white circles.

All visualisations are computed for ``num_envs_logged`` environments (taken
from the leading batch dimension of the inputs) and logged every ``interval``
environment steps.

Design notes
------------
- Forward hooks on the ReLU modules of ``_conv_trunk`` capture feature maps
  in-place whenever ``extract()`` is called.  No extra forward pass is needed.
- Hooks are only registered when at least one visualisation that needs them is
  enabled, so overhead is zero when all such visualisations are disabled.
- Tensors are captured by-reference (no copy); they are detached and moved to
  CPU only at log time.
- A dedicated :class:`~torch.utils.tensorboard.SummaryWriter` is opened at the
  provided *log_dir* — TensorBoard discovers multiple event files in the same
  directory automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from so101.utils.feature_extraction.feature_extraction import (
        CnnSpatialSoftmaxFeatureExtractor,
        ResNet18SpatialSoftmaxFeatureExtractor,
        VisionFeatureExtractor,
    )
    from so101_rl.configurations.so101_env_params import VisionDebugCfg


class VisionDebugLogger:
    """Log CNN vision debug visualisations to TensorBoard.

    Args:
        extractor: The active :class:`VisionFeatureExtractor`.  Must be an
            instance of :class:`CnnSpatialSoftmaxFeatureExtractor` for
            conv-layer and keypoint visualisations; ResNet18 skips those.
        log_dir: Directory passed to :class:`~torch.utils.tensorboard.SummaryWriter`.
        cfg: :class:`VisionDebugCfg` loaded from the YAML config.
    """

    def __init__(
        self,
        extractor: "VisionFeatureExtractor",
        log_dir: str,
        cfg: "VisionDebugCfg",
    ) -> None:
        self._cfg = cfg
        self._extractor = extractor
        self._writer = SummaryWriter(log_dir=log_dir) if cfg.enabled else None

        # Feature maps captured by forward hooks: list[Tensor (N, C, Hc, Wc)]
        # Index 0 = first conv block, -1 = last conv block.
        self._feature_maps: list[torch.Tensor] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

        if not cfg.enabled:
            return

        _needs_hooks = (
            cfg.conv_layer_maps.enabled
            or cfg.activation_heatmap.enabled
            or cfg.keypoints.enabled
        )
        if _needs_hooks:
            self._register_hooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        raw_image: torch.Tensor,
        pipelined_image: torch.Tensor,
        step: int,
    ) -> None:
        """Log all enabled visualisations for the current environment step.

        Must be called *after* :meth:`VisionFeatureExtractor.extract` so that
        the forward hooks have already fired.

        Args:
            raw_image: ``(N, H, W, 3)`` uint8 tensor from the TiledCamera.
            pipelined_image: ``(N, 3, H, W)`` float tensor after the image
                pipeline.
            step: Global environment step counter (used as the x-axis value in
                TensorBoard).
        """
        if not self._cfg.enabled:
            return
        if step % self._cfg.interval != 0:
            return

        n = self._cfg.num_envs_logged

        # Clamp to batch size
        n = min(n, raw_image.shape[0])

        raw_u8 = raw_image[:n]  # (n, H, W, 3) uint8
        pipe_f = pipelined_image[:n]  # (n, 3, H, W) float

        if self._cfg.raw_image.enabled:
            self._write_raw_image(raw_u8, step)

        if self._cfg.pipelined_image.enabled:
            self._write_pipelined_image(pipe_f, step)

        if self._feature_maps and self._cfg.conv_layer_maps.enabled:
            self._write_conv_layer_maps(n, step)

        if self._feature_maps and self._cfg.activation_heatmap.enabled:
            self._write_activation_heatmap(pipe_f, step)

        if self._feature_maps and self._cfg.keypoints.enabled:
            self._write_keypoints(pipe_f, step)

    def close(self) -> None:
        """Remove hooks and close the SummaryWriter."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if self._writer is not None:
            self._writer.close()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on the ReLU activations of the conv trunk.

        Only works for :class:`CnnSpatialSoftmaxFeatureExtractor`; silently
        skips for ResNet18.
        """
        from so101.utils.feature_extraction.feature_extraction import (
            CnnSpatialSoftmaxFeatureExtractor,
        )

        if not isinstance(self._extractor, CnnSpatialSoftmaxFeatureExtractor):
            return

        conv_trunk: nn.Sequential = self._extractor._vision_backbone
        # Conv trunk is built as Conv2d → BatchNorm2d → ReLU, repeating.
        # Hook every ReLU (index 2, 5, 8, … i.e. every 3rd module from index 2).
        for idx, module in enumerate(conv_trunk):
            if isinstance(module, nn.ReLU):
                # Capture layer index at definition time with default arg.
                def _make_hook(layer_idx: int):
                    def _hook(
                        _module: nn.Module,
                        _input: tuple,
                        output: torch.Tensor,
                    ) -> None:
                        # Grow list on first pass; overwrite on subsequent passes.
                        if layer_idx < len(self._feature_maps):
                            self._feature_maps[layer_idx] = output
                        else:
                            # Fill any gaps first (shouldn't happen in practice)
                            while len(self._feature_maps) < layer_idx:
                                self._feature_maps.append(torch.empty(0))
                            self._feature_maps.append(output)

                    return _hook

                handle = module.register_forward_hook(
                    _make_hook(len(self._hook_handles))
                )
                self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # Per-visualisation writers
    # ------------------------------------------------------------------

    def _write_raw_image(self, raw_u8: torch.Tensor, step: int) -> None:
        """Log raw uint8 camera frames as ``VisionDebug/raw_image``."""
        # raw_u8: (n, H, W, 3) uint8 → (n, 3, H, W) uint8
        imgs = raw_u8.permute(0, 3, 1, 2).float() / 255.0
        self._writer.add_images("VisionDebug/raw_image", imgs.cpu(), step)

    def _write_pipelined_image(self, pipe_f: torch.Tensor, step: int) -> None:
        """Log pipelined float frames as ``VisionDebug/pipelined_image``.

        The pipelined image may have been normalised with ImageNet statistics
        so we de-normalise to roughly [0, 1] for display purposes only.
        """
        # Normalize each image independently to [0, 1] for visibility.
        imgs = pipe_f.cpu().float()
        lo = imgs.flatten(1).min(dim=1).values[:, None, None, None]
        hi = imgs.flatten(1).max(dim=1).values[:, None, None, None]
        imgs = (imgs - lo) / (hi - lo + 1e-6)
        self._writer.add_images("VisionDebug/pipelined_image", imgs, step)

    def _write_conv_layer_maps(self, n: int, step: int) -> None:
        """Log feature-map tile grids for each conv block.

        Tags: ``VisionDebug/conv_layer_maps/layer_<i>``
        Grid layout: one row per env, each cell is a feature map channel
        (normalised to [0, 1]).
        """
        max_ch = self._cfg.conv_layer_maps.max_channels
        for layer_idx, fmap in enumerate(self._feature_maps):
            # fmap: (N, C, Hc, Wc)
            fmap_n = fmap[:n].detach().cpu().float()  # (n, C, Hc, Wc)
            C = fmap_n.shape[1]
            num_ch = min(C, max_ch)
            fmap_n = fmap_n[:, :num_ch]  # (n, num_ch, Hc, Wc)

            # Normalise each channel independently to [0, 1]
            flat = fmap_n.reshape(n, num_ch, -1)
            lo = flat.min(dim=-1).values[:, :, None, None]
            hi = flat.max(dim=-1).values[:, :, None, None]
            fmap_n = (fmap_n - lo) / (hi - lo + 1e-6)

            # Build a tile: (n*num_ch, 1, Hc, Wc) → TB interprets as greyscale
            tiles = fmap_n.reshape(n * num_ch, 1, fmap_n.shape[2], fmap_n.shape[3])
            self._writer.add_images(
                f"VisionDebug/conv_layer_maps/layer_{layer_idx}",
                tiles,
                step,
            )

    def _write_activation_heatmap(self, pipe_f: torch.Tensor, step: int) -> None:
        """Log the mean-activation heatmap of the final conv layer.

        The per-channel mean activation of the deepest feature map is upsampled
        to match the pipelined image resolution, mapped to a red colour scale,
        and alpha-blended (50 %) over the pipelined image.

        Tag: ``VisionDebug/activation_heatmap``
        """
        last_fmap = self._feature_maps[-1].detach().cpu().float()  # (N, C, Hc, Wc)
        n = pipe_f.shape[0]
        last_fmap = last_fmap[:n]

        # Mean over channels → (n, 1, Hc, Wc)
        heat = last_fmap.mean(dim=1, keepdim=True)

        # Upsample to pipelined image resolution
        _, _, H, W = pipe_f.shape
        heat = F.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)

        # Normalise to [0, 1] per image
        flat = heat.reshape(n, -1)
        lo = flat.min(dim=1).values[:, None, None, None]
        hi = flat.max(dim=1).values[:, None, None, None]
        heat = (heat - lo) / (hi - lo + 1e-6)  # (n, 1, H, W)

        # Red channel only: R=heat, G=0, B=0
        red_overlay = torch.cat(
            [heat, torch.zeros_like(heat), torch.zeros_like(heat)], dim=1
        )  # (n, 3, H, W)

        # Normalise the pipelined image to [0, 1] for blending
        bg = pipe_f.cpu().float()
        bg_flat = bg.flatten(1)
        lo_bg = bg_flat.min(dim=1).values[:, None, None, None]
        hi_bg = bg_flat.max(dim=1).values[:, None, None, None]
        bg = (bg - lo_bg) / (hi_bg - lo_bg + 1e-6)

        blended = 0.5 * bg + 0.5 * red_overlay
        blended = blended.clamp(0.0, 1.0)

        self._writer.add_images("VisionDebug/activation_heatmap", blended, step)

    def _write_keypoints(self, pipe_f: torch.Tensor, step: int) -> None:
        """Draw SpatialSoftmax keypoints over the pipelined image.

        Each channel of the final conv feature map contributes one (x, y)
        keypoint via SpatialSoftmax.  Keypoints are drawn as filled circles
        (radius 2 px) in white over the normalised pipelined image.

        Tag: ``VisionDebug/keypoints``
        """
        from so101.utils.feature_extraction.spatial_softmax import SpatialSoftmax

        last_fmap = self._feature_maps[-1].detach()  # keep on GPU if possible
        n = pipe_f.shape[0]
        last_fmap = last_fmap[:n]

        # Recompute SpatialSoftmax coordinates from stored feature map
        ss = SpatialSoftmax().to(last_fmap.device)
        with torch.no_grad():
            coords = ss(last_fmap)  # (n, 2*C); first C: x coords, next C: y coords

        C = last_fmap.shape[1]
        kp_x = coords[:, :C].cpu().float()  # (n, C) in [-1, 1]
        kp_y = coords[:, C:].cpu().float()  # (n, C) in [-1, 1]

        _, _, H, W = pipe_f.shape

        # Un-normalise to pixel space
        px = ((kp_x + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)  # (n, C)
        py = ((kp_y + 1.0) * 0.5 * (H - 1)).long().clamp(0, H - 1)  # (n, C)

        # Normalise pipelined image to [0, 1] for display
        bg = pipe_f.cpu().float()
        bg_flat = bg.flatten(1)
        lo = bg_flat.min(dim=1).values[:, None, None, None]
        hi = bg_flat.max(dim=1).values[:, None, None, None]
        canvas = (bg - lo) / (hi - lo + 1e-6)  # (n, 3, H, W)
        canvas = canvas.clamp(0.0, 1.0)

        # Draw filled squares (radius r) for each keypoint; pure-PyTorch rasteriser
        r = 2
        for env_i in range(n):
            for kp_i in range(C):
                cx = int(px[env_i, kp_i].item())
                cy = int(py[env_i, kp_i].item())
                x0 = max(cx - r, 0)
                x1 = min(cx + r + 1, W)
                y0 = max(cy - r, 0)
                y1 = min(cy + r + 1, H)
                canvas[env_i, :, y0:y1, x0:x1] = 1.0  # white

        self._writer.add_images("VisionDebug/keypoints", canvas, step)

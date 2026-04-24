"""Main processing loop: read frames → run backbone → render → write output."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


# Panel display dimensions (pixels)
_PANEL_H = 256
_PANEL_W = 256

# Feature-map grid: each channel tile inside the conv grid
_TILE_SIZE = 32


def run(args: argparse.Namespace) -> None:
    """Execute the full pipeline for the given CLI args.

    Seeds are set **before** any model/library initialisation, matching
    repo reproducibility requirements.

    Args:
        args: Validated parsed CLI args from ``cli.main()``.
    """
    # ── 1. Set seeds immediately ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(
        f"[vision_backbone_demo] seed={args.seed}  backbone={args.backbone}  "
        f"device={args.device}"
    )

    # ── 2. Build backbone + pipeline ─────────────────────────────────────────
    from .backbone import build_extractor

    bundle = build_extractor(args)

    pipeline_step_names = [type(s).__name__ for s in bundle.pipeline.steps]
    print(f"[vision_backbone_demo] image pipeline: {pipeline_step_names}")

    # ── 3. Open writer ───────────────────────────────────────────────────────
    from .io import VideoWriter, iter_frames

    writer = VideoWriter(path=args.output, fps=args.fps)

    # ── 4. Process frames ────────────────────────────────────────────────────
    from .composite import make_composite
    from .panels import (
        render_conv_grid,
        render_heatmap,
        render_keypoints,
        render_pipelined,
        render_raw,
    )

    frame_count = 0
    try:
        for raw_np in iter_frames(args.input):
            # raw_np: (H, W, 3) uint8

            # Build batch-1 tensor (N=1, H, W, C) on device
            raw_tensor = (
                torch.from_numpy(raw_np).unsqueeze(0).to(args.device)
            )  # (1, H, W, 3) uint8

            # Run through pipeline → (1, 3, H', W') float
            # (also triggers the forward hooks that populate bundle.feature_maps)
            with torch.no_grad():
                pipelined_tensor = bundle.pipeline.process(raw_tensor)
                bundle.extractor.extract(pipelined_tensor)

            pipe_chw = pipelined_tensor[0]  # (3, H', W')

            # ── Render panels ─────────────────────────────────────────────
            panel_raw = render_raw(raw_np)
            panel_pipe = render_pipelined(pipe_chw)

            if bundle.feature_maps:
                last_fmap = bundle.feature_maps[-1][0:1].detach()  # (1, C, Hc, Wc)
                # Ensure on CPU-compatible shape for panel functions
                panel_heat = render_heatmap(pipe_chw, last_fmap)
                panel_kp = render_keypoints(pipe_chw, last_fmap)

                conv_grids = []
                for i, fmap in enumerate(bundle.feature_maps):
                    grid = render_conv_grid(
                        fmap[0:1].detach(),
                        max_channels=args.max_channels,
                        tile_size=_TILE_SIZE,
                    )
                    conv_grids.append(grid)
            else:
                # ResNet18 hooks are registered but may yield many activations;
                # gracefully handle the case where no maps were captured yet.
                blank = np.zeros(
                    (bundle.image_size[0], bundle.image_size[1], 3), dtype=np.uint8
                )
                panel_heat = blank
                panel_kp = blank
                conv_grids = []

            # ── Composite ─────────────────────────────────────────────────
            composite = make_composite(
                raw=panel_raw,
                pipelined=panel_pipe,
                heatmap=panel_heat,
                keypoints=panel_kp,
                conv_grids=conv_grids,
                backbone_name=bundle.backbone_name,
                panel_h=_PANEL_H,
                panel_w=_PANEL_W,
            )

            writer.write(composite)
            frame_count += 1

            if frame_count % 10 == 0:
                print(
                    f"[vision_backbone_demo] processed {frame_count} frames…",
                    end="\r",
                    flush=True,
                )

    finally:
        writer.close()
        bundle.close()

    print(
        f"\n[vision_backbone_demo] done — {frame_count} frame(s) written to "
        f"{args.output}"
    )
    print(
        f"[vision_backbone_demo] backbone={args.backbone}  "
        f"conv blocks captured={len(bundle.feature_maps)}"
    )

"""Assemble per-frame panel images into a single composite frame.

Layout
------
Top row (fixed 4 panels):
    raw | pipelined | heatmap | keypoints

Bottom area (variable):
    One conv-layer grid panel per feature-map block, tiled left-to-right
    then wrapping to additional rows as needed.

All panels are resized to a common ``panel_h × panel_w`` before tiling.
The composite is annotated with a white label strip at the top of each panel.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# Label style constants
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LABEL_H = 20  # pixels reserved at top of each panel for the text strip
_LABEL_BG = (40, 40, 40)  # dark grey BGR
_LABEL_FG = (255, 255, 255)  # white BGR


def _add_label(panel: np.ndarray, text: str) -> np.ndarray:
    """Return a copy of *panel* with a dark label strip at the top."""
    h, w = panel.shape[:2]
    out = panel.copy()
    # Draw label background strip
    out[:_LABEL_H, :] = _LABEL_BG
    # Put text — OpenCV works in BGR; our panels are RGB so colour is fine
    # (the label is greyscale anyway).
    cv2.putText(
        out,
        text,
        (4, _LABEL_H - 5),
        _FONT,
        _FONT_SCALE,
        _LABEL_FG,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return out


def _resize_panel(panel: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize to ``(target_h, target_w, 3)`` using area-mode (good for downscale)."""
    if panel.shape[0] == target_h and panel.shape[1] == target_w:
        return panel
    return cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_composite(
    raw: np.ndarray,
    pipelined: np.ndarray,
    heatmap: np.ndarray,
    keypoints: np.ndarray,
    conv_grids: list[np.ndarray],
    backbone_name: str,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    """Tile all panels into one composite image.

    Args:
        raw: ``(H, W, 3)`` uint8 raw frame panel.
        pipelined: ``(H, W, 3)`` uint8 pipelined frame panel.
        heatmap: ``(H, W, 3)`` uint8 heatmap overlay panel.
        keypoints: ``(H, W, 3)`` uint8 keypoints overlay panel.
        conv_grids: List of ``(Hg, Wg, 3)`` uint8 conv-layer grid panels,
            one per feature-map block.
        backbone_name: Used only in the conv-grid labels.
        panel_h: Target height for every panel.
        panel_w: Target width for every panel.

    Returns:
        Single ``(composite_H, composite_W, 3)`` uint8 RGB array.
    """

    def _prep(panel: np.ndarray, label: str) -> np.ndarray:
        p = _resize_panel(panel, panel_h, panel_w)
        return _add_label(p, label)

    # ── Top row: 4 fixed panels ──────────────────────────────────────────────
    top_panels = [
        _prep(raw, "raw"),
        _prep(pipelined, "pipelined"),
        _prep(heatmap, "heatmap"),
        _prep(keypoints, "keypoints"),
    ]
    top_row = np.concatenate(top_panels, axis=1)  # (panel_h, 4*panel_w, 3)

    if not conv_grids:
        return top_row

    # ── Bottom rows: conv-layer grid panels ──────────────────────────────────
    n = len(conv_grids)
    ncols = 4  # same number of columns as top row
    nrows = math.ceil(n / ncols)

    labelled_grids = [_prep(g, f"conv_block_{i}") for i, g in enumerate(conv_grids)]

    # Pad to fill the grid
    pad = nrows * ncols - n
    blank = np.full((panel_h, panel_w, 3), fill_value=20, dtype=np.uint8)
    labelled_grids += [blank] * pad

    bottom_rows_list = []
    for row_i in range(nrows):
        row_panels = labelled_grids[row_i * ncols : (row_i + 1) * ncols]
        bottom_rows_list.append(np.concatenate(row_panels, axis=1))
    bottom_rows = np.concatenate(bottom_rows_list, axis=0)

    # ── Stack top + bottom ───────────────────────────────────────────────────
    # Both have width == 4 * panel_w (guaranteed by ncols=4 above).
    composite = np.concatenate([top_row, bottom_rows], axis=0)
    return composite

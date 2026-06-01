"""overlay.py — Live OpenCV visualisation during real-robot inference."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


class OverlayRenderer:
    """Non-blocking OpenCV window displaying the camera feed and joint state.

    Shows:
    - Camera frame with frame counter
    - Commanded vs. measured joint bars (right side panel)

    Parameters
    ----------
    joint_names:
        List of joint name strings (used for labels).
    panel_width:
        Width in pixels of the joint-bar side panel.
    """

    def __init__(self, joint_names: list[str], panel_width: int = 200) -> None:
        self._joint_names = joint_names
        self._panel_width = panel_width
        self._frame_count = 0
        cv2.namedWindow("so101_real", cv2.WINDOW_NORMAL)

    def update(
        self,
        frame_rgb: np.ndarray,
        q_measured: torch.Tensor,
        q_commanded: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        """Render one frame.

        Parameters
        ----------
        frame_rgb:
            Camera frame ``(H, W, 3)`` uint8 RGB.
        q_measured:
            Measured joint positions in radians ``(n_joints,)``.
        q_commanded:
            Commanded joint targets in radians ``(n_joints,)``.
        action:
            Raw policy output in [-1, 1] ``(act_dim,)``.
        """
        self._frame_count += 1
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        H, W = frame_bgr.shape[:2]

        # ── Frame counter overlay ─────────────────────────────────────────────
        cv2.putText(
            frame_bgr,
            f"step {self._frame_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # ── Joint bar panel ───────────────────────────────────────────────────
        panel = np.zeros((H, self._panel_width, 3), dtype=np.uint8)
        n = len(self._joint_names)
        row_h = max(H // (n + 1), 20)

        q_m = q_measured.detach().cpu().float().numpy()
        q_c = q_commanded.detach().cpu().float().numpy()

        for i, name in enumerate(self._joint_names):
            y = (i + 1) * row_h
            label = name[-10:]  # truncate long names
            cv2.putText(
                panel,
                label,
                (5, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
            )
            # Measured bar (blue)
            bar_max = self._panel_width - 20
            m_norm = float(np.clip((q_m[i] + np.pi) / (2 * np.pi), 0, 1))
            c_norm = float(np.clip((q_c[i] + np.pi) / (2 * np.pi), 0, 1))
            cv2.rectangle(
                panel,
                (10, y),
                (10 + int(m_norm * bar_max), y + 6),
                (255, 100, 0),
                -1,
            )
            # Commanded bar (green)
            cv2.rectangle(
                panel,
                (10, y + 8),
                (10 + int(c_norm * bar_max), y + 14),
                (0, 255, 100),
                -1,
            )

        combined = np.concatenate([frame_bgr, panel], axis=1)
        cv2.imshow("so101_real", combined)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow("so101_real")

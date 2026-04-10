"""Multi-task loss functions and evaluation metrics for CNN backbone pretraining.

Loss composition
----------------
.. math::

    L = w_{\\text{pos}}   \\cdot L_{\\text{pos}}
      + w_{\\text{align}} \\cdot L_{\\text{align}}
      + w_{\\text{rot}}   \\cdot L_{\\text{rot}}
      + w_{\\text{ht}}    \\cdot L_{\\text{ht}}
      + w_{\\text{vis}}   \\cdot L_{\\text{vis}}

where:

- :math:`L_{\\text{pos}}`   — Huber loss between predicted and target
  ``cube_pos_gz`` (normalised 3-D position in grip-zone frame).
- :math:`L_{\\text{align}}` — MSE between predicted and target
  ``gripper_cube_alignment`` scalar in ``[-1, 1]``.
- :math:`L_{\\text{rot}}`   — MSE between predicted and target
  ``cube_rot6d_gz`` rotation6D vector.
- :math:`L_{\\text{ht}}`    — Huber loss between predicted and target
  ``cube_height_w`` (normalised height above resting position).
- :math:`L_{\\text{vis}}`   — binary cross-entropy with logits against the
  ``cube_in_camera_frame`` target (float 0 or 1).

All losses are mean-reduced over the batch.
The Huber delta is controlled by ``loss_cfg["huber_delta"]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_multitask_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Compute the weighted multi-task loss.

    Args:
        predictions: Output of :meth:`MultiTaskCnn.forward`.
        targets: Batch of target dicts from :class:`TelemetryDataset`.
        loss_cfg: The ``loss`` section of ``cnn_pretrain.yaml``.

    Returns:
        Dict with keys ``total``, ``cube_pos_gz``, ``gripper_cube_alignment``,
        ``cube_rot6d_gz``, ``cube_height_w``, ``cube_in_camera_frame``; all are
        scalar ``Tensor`` values with gradients.
    """
    for key in (
        "weight_cube_pos_gz",
        "weight_gripper_cube_alignment",
        "weight_cube_rot6d_gz",
        "weight_cube_height_w",
        "weight_cube_in_camera_frame",
        "huber_delta",
    ):
        if key not in loss_cfg or loss_cfg[key] is None:
            raise ValueError(
                f"loss_cfg['{key}'] must be set explicitly; no default is allowed."
            )

    w_pos = float(loss_cfg["weight_cube_pos_gz"])
    w_align = float(loss_cfg["weight_gripper_cube_alignment"])
    w_rot = float(loss_cfg["weight_cube_rot6d_gz"])
    w_ht = float(loss_cfg["weight_cube_height_w"])
    w_vis = float(loss_cfg["weight_cube_in_camera_frame"])
    delta = float(loss_cfg["huber_delta"])

    # ── cube_pos_gz: Huber (dim 3, normalised metres) ──
    pos_pred = predictions["cube_pos_gz_pred"]  # (N, 3)
    pos_target = targets["cube_pos_gz"]  # (N, 3)
    pos_loss = F.huber_loss(pos_pred, pos_target, delta=delta)

    # ── gripper_cube_alignment: MSE (scalar in [-1, 1]) ──
    align_pred = predictions["gripper_cube_alignment_pred"].squeeze(-1)  # (N,)
    align_target = targets["gripper_cube_alignment"]  # (N,)
    align_loss = F.mse_loss(align_pred, align_target)

    # ── cube_rot6d_gz: MSE (rotation6D in [-1, 1]) ──
    rot_pred = predictions["cube_rot6d_gz_pred"]  # (N, 6)
    rot_target = targets["cube_rot6d_gz"]  # (N, 6)
    rot_loss = F.mse_loss(rot_pred, rot_target)

    # ── cube_height_w: Huber (scalar, normalised metres) ──
    ht_pred = predictions["cube_height_w_pred"].squeeze(-1)  # (N,)
    ht_target = targets["cube_height_w"]  # (N,)
    ht_loss = F.huber_loss(ht_pred, ht_target, delta=delta)

    # ── cube_in_camera_frame: BCE with logits ──
    vis_logit = predictions["cube_in_camera_frame_logit"].squeeze(-1)  # (N,)
    vis_target = targets["cube_in_camera_frame"]  # (N,)
    vis_loss = F.binary_cross_entropy_with_logits(vis_logit, vis_target)

    total = (
        w_pos * pos_loss
        + w_align * align_loss
        + w_rot * rot_loss
        + w_ht * ht_loss
        + w_vis * vis_loss
    )
    return {
        "total": total,
        "cube_pos_gz": pos_loss,
        "gripper_cube_alignment": align_loss,
        "cube_rot6d_gz": rot_loss,
        "cube_height_w": ht_loss,
        "cube_in_camera_frame": vis_loss,
    }


def compute_multitask_metrics(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute per-head evaluation metrics.

    No gradients are required; the function runs inside ``torch.no_grad()``.

    Returns:
        Dict with keys:

        - ``cube_pos_gz_mae``            — mean L1 distance in normalised units.
        - ``gripper_cube_alignment_mae`` — mean absolute error of alignment scalar.
        - ``cube_rot6d_gz_mse``          — MSE of rotation6D prediction.
        - ``cube_height_w_mae``          — mean absolute error in normalised units.
        - ``cube_in_camera_frame_acc``   — binary accuracy of visibility logit.
    """
    with torch.no_grad():
        pos_mae = (
            (predictions["cube_pos_gz_pred"] - targets["cube_pos_gz"])
            .abs()
            .mean()
            .item()
        )

        align_mae = (
            (
                predictions["gripper_cube_alignment_pred"].squeeze(-1)
                - targets["gripper_cube_alignment"]
            )
            .abs()
            .mean()
            .item()
        )

        rot_mse = F.mse_loss(
            predictions["cube_rot6d_gz_pred"], targets["cube_rot6d_gz"]
        ).item()

        ht_mae = (
            (predictions["cube_height_w_pred"].squeeze(-1) - targets["cube_height_w"])
            .abs()
            .mean()
            .item()
        )

        vis_pred = (
            predictions["cube_in_camera_frame_logit"].squeeze(-1) >= 0.0
        ).float()
        vis_acc = (vis_pred == targets["cube_in_camera_frame"]).float().mean().item()

    return {
        "cube_pos_gz_mae": pos_mae,
        "gripper_cube_alignment_mae": align_mae,
        "cube_rot6d_gz_mse": rot_mse,
        "cube_height_w_mae": ht_mae,
        "cube_in_camera_frame_acc": vis_acc,
    }

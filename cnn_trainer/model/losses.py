"""Multi-task loss functions and evaluation metrics for CNN backbone pretraining.

Loss composition
----------------
.. math::

    L = w_{\\text{grip}} \\cdot L_{\\text{grip}}
      + w_{\\text{ori}}  \\cdot L_{\\text{ori}}
      + w_{\\text{vis}}  \\cdot L_{\\text{vis}}

where:

- :math:`L_{\\text{grip}}` — binary cross-entropy with logits against the
  ``is_cube_in_grip_position`` target (float 0 or 1).
- :math:`L_{\\text{ori}}`  — MSE between the L2-normalised predicted
  quaternion and the unit-quaternion target ``cube_quat_gripzone_wxyz``.
  Normalisation is controlled by ``loss_cfg["orientation_normalization"]``.
- :math:`L_{\\text{vis}}`  — binary cross-entropy with logits against the
  ``cube_in_camera_frame`` target (float 0 or 1).

All losses are mean-reduced over the batch.
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
        predictions: Output of :meth:`PretrainCnn.forward`.
        targets: Batch of target dicts from :class:`TelemetryDataset`.
        loss_cfg: The ``loss`` section of ``cnn_pretrain.yaml``.

    Returns:
        Dict with keys ``total``, ``grip_position``, ``orientation``,
        ``visibility``; all are scalar ``Tensor`` values with gradients.
    """
    w_grip = float(loss_cfg["weight_grip_position"])
    w_orientation = float(loss_cfg["weight_orientation"])
    w_visibility = float(loss_cfg["weight_visibility"])
    normalize_quat = bool(loss_cfg.get("orientation_normalization", True))

    # ── Grip position: BCE with logits ──
    grip_logit = predictions["grip_position_logit"].squeeze(-1)  # (N,)
    grip_target = targets["is_cube_in_grip_position"]  # (N,)
    grip_loss = F.binary_cross_entropy_with_logits(grip_logit, grip_target)

    # ── Orientation: MSE on (optionally normalised) quaternion ──
    orientation_pred = predictions["orientation_pred"]  # (N, 4)
    if normalize_quat:
        orientation_pred = F.normalize(orientation_pred, p=2, dim=-1)
    orientation_target = targets["cube_quat_gripzone_wxyz"]  # (N, 4)
    orientation_loss = F.mse_loss(orientation_pred, orientation_target)

    # ── Visibility: BCE with logits ──
    vis_logit = predictions["visibility_logit"].squeeze(-1)  # (N,)
    vis_target = targets["cube_in_camera_frame"]  # (N,)
    visibility_loss = F.binary_cross_entropy_with_logits(vis_logit, vis_target)

    total = (
        w_grip * grip_loss
        + w_orientation * orientation_loss
        + w_visibility * visibility_loss
    )
    return {
        "total": total,
        "grip_position": grip_loss,
        "orientation": orientation_loss,
        "visibility": visibility_loss,
    }


def compute_multitask_metrics(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute per-head evaluation metrics.

    No gradients are required; the function runs inside ``torch.no_grad()``.

    Returns:
        Dict with keys:

        - ``grip_position_acc`` — binary accuracy of grip-position prediction.
        - ``visibility_acc``    — binary accuracy of visibility prediction.
        - ``orientation_mse``   — MSE between normalised predicted quaternion
          and target quaternion.
    """
    with torch.no_grad():
        # Binary accuracy: threshold logit at 0 (equivalent to sigmoid > 0.5)
        grip_pred = (predictions["grip_position_logit"].squeeze(-1) >= 0.0).float()
        grip_acc = (
            (grip_pred == targets["is_cube_in_grip_position"]).float().mean().item()
        )

        vis_pred = (predictions["visibility_logit"].squeeze(-1) >= 0.0).float()
        vis_acc = (vis_pred == targets["cube_in_camera_frame"]).float().mean().item()

        orientation_pred = F.normalize(predictions["orientation_pred"], p=2, dim=-1)
        orientation_mse = F.mse_loss(
            orientation_pred, targets["cube_quat_gripzone_wxyz"]
        ).item()

    return {
        "grip_position_acc": grip_acc,
        "visibility_acc": vis_acc,
        "orientation_mse": orientation_mse,
    }

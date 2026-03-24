"""Multi-task PretrainCnn model for CNN backbone pretraining.

Wraps :class:`TrainableCnnFeatureExtractor` (from ``so101_utils``) with three
task-specific prediction heads that push the backbone to learn domain-relevant
visual features::

    image (C, H, W)
        ↓
    CNN conv trunk
        ↓
    SpatialSoftmax
        ↓
    Projection MLP   ← shared latent (output_dim,)
        ↓
        ├── GripPositionHead  → logit  (1,)  [BCE with is_cube_in_grip_position]
        ├── OrientationHead   → quat   (4,)  [MSE with cube_quat_gripzone_wxyz]
        └── VisibilityHead    → logit  (1,)  [BCE with cube_in_camera_frame]

After supervised pretraining, only the backbone weights are extracted via
:meth:`PretrainCnn.backbone_state_dict` and loaded into the RL policy's
``_cnn`` module.  The heads are discarded.

Architecture compatibility note
--------------------------------
``backbone_cfg`` must use the same values as ``models.policy.cnn`` in
``skrl_ppo_cfg.yaml`` so that the saved backbone weights load cleanly into
:class:`TrainableCnnFeatureExtractor` inside the RL actor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from so101_utils.feature_extraction.feature_extraction import (
    TrainableCnnFeatureExtractor,
)


# ── Head builder ──────────────────────────────────────────────────────────────


def _build_head(
    input_dim: int, hidden_dims: list[int], output_dim: int
) -> nn.Sequential:
    """MLP head: ``[Linear → ReLU]* → Linear``."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ── Config validation helper ──────────────────────────────────────────────────


def _require_keys(d: dict, keys: list[str], section: str) -> None:
    for k in keys:
        if k not in d or d[k] is None:
            raise ValueError(
                f"{section}.{k} must be set explicitly; no default is allowed."
            )


# ── Model ─────────────────────────────────────────────────────────────────────


class PretrainCnn(nn.Module):
    """Multi-task CNN backbone with three supervised prediction heads.

    Args:
        backbone_cfg: Dict matching :class:`TrainableCnnFeatureExtractor`'s
            constructor keys: ``in_channels``, ``channels``, ``kernel_sizes``,
            ``strides``, ``mlp_hidden_dims``, ``output_dim``.
        heads_cfg: Dict with keys ``grip_position``, ``orientation``,
            ``visibility``; each must contain a ``hidden_dims`` list.

    Example config (from ``cnn_pretrain.yaml``)::

        backbone:
          in_channels: 3
          channels:       [32, 64, 128, 128]
          kernel_sizes:   [5,  4,  3,   3]
          strides:        [2,  2,  2,   1]
          mlp_hidden_dims: [256]
          output_dim: 256

        heads:
          grip_position: {hidden_dims: [128]}
          orientation:   {hidden_dims: [128]}
          visibility:    {hidden_dims: []}
    """

    def __init__(self, backbone_cfg: dict, heads_cfg: dict) -> None:
        super().__init__()
        _require_keys(
            backbone_cfg,
            [
                "in_channels",
                "channels",
                "kernel_sizes",
                "strides",
                "mlp_hidden_dims",
                "output_dim",
            ],
            section="backbone",
        )
        _require_keys(
            heads_cfg,
            ["grip_position", "orientation", "visibility"],
            section="heads",
        )
        for name in ("grip_position", "orientation", "visibility"):
            if "hidden_dims" not in heads_cfg[name]:
                raise ValueError(f"heads.{name}.hidden_dims must be set explicitly.")

        output_dim = int(backbone_cfg["output_dim"])

        self.backbone = TrainableCnnFeatureExtractor(
            in_channels=int(backbone_cfg["in_channels"]),
            channels=list(backbone_cfg["channels"]),
            kernel_sizes=list(backbone_cfg["kernel_sizes"]),
            strides=list(backbone_cfg["strides"]),
            mlp_hidden_dims=list(backbone_cfg["mlp_hidden_dims"]),
            output_dim=output_dim,
        )

        self.grip_position_head = _build_head(
            output_dim,
            list(heads_cfg["grip_position"]["hidden_dims"]),
            output_dim=1,
        )
        self.orientation_head = _build_head(
            output_dim,
            list(heads_cfg["orientation"]["hidden_dims"]),
            output_dim=4,
        )
        self.visibility_head = _build_head(
            output_dim,
            list(heads_cfg["visibility"]["hidden_dims"]),
            output_dim=1,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: ``(N, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            Dict with keys:

            - ``grip_position_logit`` — ``(N, 1)`` raw BCE logit.
            - ``orientation_pred``    — ``(N, 4)`` raw quaternion prediction.
            - ``visibility_logit``    — ``(N, 1)`` raw BCE logit.
        """
        latent = self.backbone(images)  # (N, output_dim)
        return {
            "grip_position_logit": self.grip_position_head(latent),  # (N, 1)
            "orientation_pred": self.orientation_head(latent),  # (N, 4)
            "visibility_logit": self.visibility_head(latent),  # (N, 1)
        }

    def backbone_state_dict(self) -> dict[str, Any]:
        """Return only the backbone weights.

        The returned dict is a valid ``state_dict`` for
        :class:`TrainableCnnFeatureExtractor` and can be loaded directly::

            cnn_module = TrainableCnnFeatureExtractor(...)
            cnn_module.load_state_dict(pretrain_model.backbone_state_dict())

        This is used by ``train.py`` to save ``best_backbone.pt`` /
        ``final_backbone.pt`` and by the RL training script when
        ``--cnn-backbone-checkpoint`` is specified.
        """
        prefix = "backbone."
        return {
            k[len(prefix) :]: v
            for k, v in self.state_dict().items()
            if k.startswith(prefix)
        }

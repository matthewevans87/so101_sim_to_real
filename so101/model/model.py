"""Multi-task CNN model for supervised backbone pretraining.

:class:`MultiTaskCnn` contains a CNN backbone (conv trunk → SpatialSoftmax →
projection MLP) plus five task-specific prediction heads::

    image (C, H, W)
        ↓
    CNN conv trunk
        ↓
    SpatialSoftmax
        ↓
    Projection MLP   ← shared latent (output_dim,)
        ↓
        ├── CubePosGzHead            → (3,) normalised position  [Huber with cube_pos_gz]
        ├── GripperCubeAlignmentHead → (1,) alignment scalar     [MSE  with gripper_cube_alignment]
        ├── CubeRot6dGzHead          → (6,) rotation6D           [MSE  with cube_rot6d_gz]
        ├── CubeHeightWHead          → (1,) normalised height     [Huber with cube_height_w]
        └── CubeInCameraFrameHead    → (1,) visibility logit      [BCE  with cube_in_camera_frame]

After supervised pretraining, :meth:`MultiTaskCnn.backbone_state_dict` returns
only the backbone weights.  :class:`~so101.utils.feature_extraction.feature_extraction.CnnSpatialSoftmaxFeatureExtractor`
loads a :class:`MultiTaskCnn`, truncates the projection MLP and heads, and
feeds the final conv layer output through a fresh SpatialSoftmax.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from so101.utils.feature_extraction.spatial_softmax import SpatialSoftmax


# ── Backbone ──────────────────────────────────────────────────────────────────


class _Backbone(nn.Module):
    """CNN backbone: configurable conv trunk → SpatialSoftmax → MLP projection."""

    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_sizes: list,
        strides: list,
        mlp_hidden_dims: list,
        output_dim: int,
    ) -> None:
        super().__init__()

        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                f"channels, kernel_sizes, and strides must have equal length; "
                f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}."
            )

        self._output_dim = output_dim

        # Conv trunk: Conv2d → BatchNorm2d → ReLU blocks
        conv_layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=k,
                        stride=s,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self._conv_trunk = nn.Sequential(*conv_layers)

        self._spatial_softmax = SpatialSoftmax()

        # MLP projection: SpatialSoftmax outputs 2 * channels[-1] features
        spatial_dim = 2 * channels[-1]
        mlp_layers: list = []
        prev = spatial_dim
        for h in mlp_hidden_dims:
            mlp_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        mlp_layers.append(nn.Linear(prev, output_dim))
        self._mlp = nn.Sequential(*mlp_layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        conv_feats = self._conv_trunk(images)  # (N, channels[-1], Hc, Wc)
        spatial_feats = self._spatial_softmax(conv_feats)  # (N, 2*channels[-1])
        return self._mlp(spatial_feats)  # (N, output_dim)


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


class MultiTaskCnn(nn.Module):
    """Multi-task CNN backbone with five supervised prediction heads.

    Args:
        backbone_cfg: Dict with keys ``in_channels``, ``channels``,
            ``kernel_sizes``, ``strides``, ``mlp_hidden_dims``, ``output_dim``.
        heads_cfg: Dict with keys ``cube_pos_gz``, ``gripper_cube_alignment``,
            ``cube_rot6d_gz``, ``cube_height_w``, ``cube_in_camera_frame``;
            each must contain a ``hidden_dims`` list.
            Pass ``None`` when only the backbone is needed (e.g. inside
            :class:`~so101.utils.feature_extraction.feature_extraction.CnnSpatialSoftmaxFeatureExtractor`).

    Example config (from ``cnn_pretrain.yaml``)::

        backbone:
          in_channels: 3
          channels:       [32, 64, 128, 128]
          kernel_sizes:   [5,  4,  3,   3]
          strides:        [2,  2,  2,   1]
          mlp_hidden_dims: [256]
          output_dim: 256

        heads:
          cube_pos_gz:             {hidden_dims: [128]}
          gripper_cube_alignment:  {hidden_dims: []}
          cube_rot6d_gz:           {hidden_dims: [128]}
          cube_height_w:           {hidden_dims: []}
          cube_in_camera_frame:    {hidden_dims: []}
    """

    def __init__(self, backbone_cfg: dict, heads_cfg: dict | None = None) -> None:
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

        output_dim = int(backbone_cfg["output_dim"])

        self.backbone = _Backbone(
            in_channels=int(backbone_cfg["in_channels"]),
            channels=list(backbone_cfg["channels"]),
            kernel_sizes=list(backbone_cfg["kernel_sizes"]),
            strides=list(backbone_cfg["strides"]),
            mlp_hidden_dims=list(backbone_cfg["mlp_hidden_dims"]),
            output_dim=output_dim,
        )

        if heads_cfg is not None:
            _required_head_keys = [
                "cube_pos_gz",
                "gripper_cube_alignment",
                "cube_rot6d_gz",
                "cube_height_w",
                "cube_in_camera_frame",
            ]
            _require_keys(heads_cfg, _required_head_keys, section="heads")
            for name in _required_head_keys:
                if "hidden_dims" not in heads_cfg[name]:
                    raise ValueError(
                        f"heads.{name}.hidden_dims must be set explicitly."
                    )

            self.cube_pos_gz_head = _build_head(
                output_dim,
                list(heads_cfg["cube_pos_gz"]["hidden_dims"]),
                output_dim=3,
            )
            self.gripper_cube_alignment_head = _build_head(
                output_dim,
                list(heads_cfg["gripper_cube_alignment"]["hidden_dims"]),
                output_dim=1,
            )
            self.cube_rot6d_gz_head = _build_head(
                output_dim,
                list(heads_cfg["cube_rot6d_gz"]["hidden_dims"]),
                output_dim=6,
            )
            self.cube_height_w_head = _build_head(
                output_dim,
                list(heads_cfg["cube_height_w"]["hidden_dims"]),
                output_dim=1,
            )
            self.cube_in_camera_frame_head = _build_head(
                output_dim,
                list(heads_cfg["cube_in_camera_frame"]["hidden_dims"]),
                output_dim=1,
            )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: ``(N, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            Dict with keys:

            - ``cube_pos_gz_pred``            — ``(N, 3)`` normalised position.
            - ``gripper_cube_alignment_pred`` — ``(N, 1)`` alignment scalar.
            - ``cube_rot6d_gz_pred``          — ``(N, 6)`` rotation6D.
            - ``cube_height_w_pred``          — ``(N, 1)`` normalised height.
            - ``cube_in_camera_frame_logit``  — ``(N, 1)`` raw BCE logit.
        """
        if not hasattr(self, "cube_pos_gz_head"):
            raise RuntimeError(
                "MultiTaskCnn was constructed without heads_cfg; "
                "forward() is not available."
            )
        latent = self.backbone(images)  # (N, output_dim)
        return {
            "cube_pos_gz_pred": self.cube_pos_gz_head(latent),  # (N, 3)
            "gripper_cube_alignment_pred": self.gripper_cube_alignment_head(
                latent
            ),  # (N, 1)
            "cube_rot6d_gz_pred": self.cube_rot6d_gz_head(latent),  # (N, 6)
            "cube_height_w_pred": self.cube_height_w_head(latent),  # (N, 1)
            "cube_in_camera_frame_logit": self.cube_in_camera_frame_head(
                latent
            ),  # (N, 1)
        }

    def backbone_state_dict(self) -> dict[str, Any]:
        """Return only the backbone weights.

        The returned dict can be loaded into the ``backbone`` submodule of a
        new :class:`MultiTaskCnn` instance (e.g. inside
        :class:`~so101.utils.feature_extraction.feature_extraction.CnnSpatialSoftmaxFeatureExtractor`)::

            model = MultiTaskCnn(backbone_cfg=..., heads_cfg=None)
            model.backbone.load_state_dict(pretrain_model.backbone_state_dict())

        This is used by the RL training script when ``--cnn_checkpoint`` is specified.
        """
        prefix = "backbone."
        return {
            k[len(prefix) :]: v
            for k, v in self.state_dict().items()
            if k.startswith(prefix)
        }


# ── Factory ───────────────────────────────────────────────────────────────────


def multitask_cnn_from_checkpoint(
    path: str,
    backbone_cfg: dict,
    device: str = "cuda",
) -> MultiTaskCnn:
    """Instantiate a :class:`MultiTaskCnn` (backbone only) and load weights from *path*.

    Analogous to ``torchvision.models.resnet18(weights=...)``.  The checkpoint
    may be either:

    - A **full model** state dict (keys prefixed with ``"backbone."`` plus head
      keys) — produced by ``train_cnn.py``'s ``best_model.pt`` / ``final_model.pt``.
    - A **backbone-only** state dict (bare keys) — legacy format.

    The function strips the ``"backbone."`` prefix when present and loads only
    those weights into ``model.backbone``.  Head weights are silently ignored.

    Args:
        path: Absolute or relative path to the ``.pt`` checkpoint file.
        backbone_cfg: Dict passed verbatim to :class:`MultiTaskCnn`.
        device: PyTorch device string for ``torch.load(map_location=...)``.

    Returns:
        A :class:`MultiTaskCnn` with ``heads_cfg=None`` and backbone weights loaded.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the state dict is missing keys required by the backbone.
    """
    resolved = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"MultiTaskCnn checkpoint not found: {resolved}")

    model = MultiTaskCnn(backbone_cfg=backbone_cfg, heads_cfg=None)
    state: dict[str, torch.Tensor] = torch.load(
        resolved, map_location=device, weights_only=True
    )

    # Detect full-model vs backbone-only state dict.
    if any(k.startswith("backbone.") for k in state):
        backbone_state = {
            k[len("backbone.") :]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
    else:
        backbone_state = state

    model_state = model.backbone.state_dict()
    missing = set(model_state.keys()) - set(backbone_state.keys())
    shape_errors = {
        k
        for k in set(backbone_state.keys()) & set(model_state.keys())
        if backbone_state[k].shape != model_state[k].shape
    }
    if missing:
        raise ValueError(
            f"Checkpoint is missing backbone keys: {sorted(missing)}.\n"
            "Ensure backbone_cfg matches the architecture used during pretraining."
        )
    if shape_errors:
        raise ValueError(
            f"Checkpoint has shape mismatches for backbone keys: {sorted(shape_errors)}.\n"
            "Ensure backbone_cfg matches the architecture used during pretraining."
        )

    model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[MultiTaskCnn] Loaded backbone weights from: {resolved}")
    return model

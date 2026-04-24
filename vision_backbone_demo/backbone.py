"""Build vision feature extractors + image pipelines for the demo tool.

Mirrors the backbone + pipeline construction in ``so101_lift_cube_env.py``
lines 287-328 exactly, but without any Isaac Lab / gym dependencies.

Public API
----------
``build_extractor(args)``
    Returns a ``BackboneBundle`` with the extractor, image pipeline,
    and a list that is filled with per-block feature maps after each
    ``extractor.extract()`` call (via forward hooks on each ``nn.ReLU``
    inside ``extractor._vision_backbone``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class BackboneBundle:
    """Everything needed to run one frame through the backbone and collect visuals.

    Attributes:
        extractor: The frozen :class:`VisionFeatureExtractor`.
        pipeline: Configured :class:`ImagePipeline` (uint8 HWC → float CHW).
        feature_maps: List populated by forward hooks after each ``extract()``
            call.  Index 0 = first conv-ReLU output, -1 = last.
        image_size: ``(H, W)`` that the pipeline resizes to (display hint).
        backbone_name: Human-readable name for panel labels.
        _hook_handles: Internal list; caller can ignore.
    """

    extractor: Any  # VisionFeatureExtractor
    pipeline: Any  # ImagePipeline
    feature_maps: list[torch.Tensor] = field(default_factory=list)
    image_size: tuple[int, int] = (128, 128)
    backbone_name: str = ""
    _hook_handles: list[Any] = field(default_factory=list)

    def close(self) -> None:
        """Remove all forward hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def build_extractor(args: argparse.Namespace) -> BackboneBundle:
    """Instantiate the chosen backbone, image pipeline, and hook up feature captures.

    Args:
        args: Parsed CLI args.  Must have ``.backbone``, ``.device``,
              and for ``frozen_cnn``: ``.cnn_checkpoint``, ``.backbone_cfg``.

    Returns:
        A populated :class:`BackboneBundle`.
    """
    from so101.utils.feature_extraction.feature_extraction import (
        CnnSpatialSoftmaxFeatureExtractor,
        ResNet18SpatialSoftmaxFeatureExtractor,
    )
    from so101.utils.image_processing.image_pipeline import (
        ClampPipelineStep,
        ImageNetNormalizationPipelineStep,
        ImagePipeline,
        ResizePipelineStep,
        Uint8ToFloatCHWPipelineStep,
    )

    device = args.device
    bundle = BackboneBundle(extractor=None, pipeline=None)  # type: ignore[arg-type]
    bundle.backbone_name = args.backbone

    # ── Pipeline always starts with uint8 → float CHW conversion ────────────
    steps = [Uint8ToFloatCHWPipelineStep()]

    if args.backbone == "frozen_resnet18":
        # 224×224 matches ImageNet pretraining resolution for best feature quality.
        steps.append(ResizePipelineStep((224, 224)))
        steps.append(ImageNetNormalizationPipelineStep())
        steps.append(ClampPipelineStep())
        bundle.image_size = (224, 224)

        extractor = ResNet18SpatialSoftmaxFeatureExtractor(device=device)

    elif args.backbone == "frozen_cnn":
        import yaml

        with open(args.backbone_cfg) as f:
            cfg = yaml.safe_load(f)

        _require_keys(
            cfg,
            [
                "channels",
                "kernel_sizes",
                "strides",
                "mlp_hidden_dims",
                "output_dim",
                "image_height",
                "image_width",
            ],
            section="backbone_cfg",
        )

        h, w = int(cfg["image_height"]), int(cfg["image_width"])
        steps.append(ResizePipelineStep((h, w)))
        steps.append(ClampPipelineStep())
        bundle.image_size = (h, w)

        from so101.model.model import MultiTaskCnn, multitask_cnn_from_checkpoint

        backbone_cfg = {
            "in_channels": 3,
            "channels": list(cfg["channels"]),
            "kernel_sizes": list(cfg["kernel_sizes"]),
            "strides": list(cfg["strides"]),
            "mlp_hidden_dims": list(cfg["mlp_hidden_dims"]),
            "output_dim": int(cfg["output_dim"]),
        }
        model = multitask_cnn_from_checkpoint(
            path=args.cnn_checkpoint,
            backbone_cfg=backbone_cfg,
            device=device,
        )
        extractor = CnnSpatialSoftmaxFeatureExtractor(model=model, device=device)

    else:
        raise ValueError(
            f"Unknown backbone {args.backbone!r}. "
            "Must be 'frozen_resnet18' or 'frozen_cnn'."
        )

    bundle.extractor = extractor
    bundle.pipeline = ImagePipeline(steps)

    _register_hooks(extractor, bundle)

    return bundle


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def _register_hooks(extractor: Any, bundle: BackboneBundle) -> None:
    """Register forward hooks on every nn.ReLU inside ``extractor._vision_backbone``.

    Works for both :class:`ResNet18SpatialSoftmaxFeatureExtractor` and
    :class:`CnnSpatialSoftmaxFeatureExtractor`.

    For the custom CNN the trunk is a flat ``nn.Sequential`` of
    Conv2d → BatchNorm2d → ReLU blocks, so each ReLU corresponds to one
    conv block — matching the convention in ``VisionDebugLogger._register_hooks``.

    For ResNet18 the trunk is a deep nested structure; we walk it recursively
    and hook every ``nn.ReLU``, giving one feature map per activation in the
    network (many maps, but all meaningful at display time).
    """
    feature_maps: list[torch.Tensor] = bundle.feature_maps
    hook_handles: list[Any] = bundle._hook_handles

    def _make_hook(layer_idx: int):
        def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
            if layer_idx < len(feature_maps):
                feature_maps[layer_idx] = output
            else:
                while len(feature_maps) < layer_idx:
                    feature_maps.append(torch.empty(0))
                feature_maps.append(output)

        return _hook

    conv_trunk: nn.Module = extractor._vision_backbone
    relu_count = 0
    for module in conv_trunk.modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(_make_hook(relu_count))
            hook_handles.append(handle)
            relu_count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_keys(d: dict, keys: list[str], section: str) -> None:
    for k in keys:
        if k not in d or d[k] is None:
            raise ValueError(
                f"{section}.{k} must be set explicitly; no default is allowed."
            )

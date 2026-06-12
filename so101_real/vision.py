"""vision.py — Build the frozen vision feature extractor from a deploy bundle."""

from __future__ import annotations

import torch

from so101.utils.feature_extraction.feature_extraction import (
    CnnSpatialSoftmaxFeatureExtractor,
    ResNet18SpatialSoftmaxFeatureExtractor,
)

from .bundle import DeployBundle


def build_vision_encoder(
    bundle: DeployBundle,
    device: str | torch.device = "cpu",
) -> ResNet18SpatialSoftmaxFeatureExtractor | CnnSpatialSoftmaxFeatureExtractor:
    """Instantiate the frozen vision encoder described in the bundle manifest.

    Parameters
    ----------
    bundle:
        Validated DeployBundle.
    device:
        Target device for inference.

    Returns
    -------
    Feature extractor with an ``.extract(images)`` method.
    ``images`` shape: ``(N, C, H, W)`` float32 in the range expected by the
    pipeline (after running through ``build_deploy_pipeline``).
    """
    encoder_type = bundle.encoder_type

    if encoder_type == "frozen_resnet18":
        return ResNet18SpatialSoftmaxFeatureExtractor(device=device)

    if encoder_type == "frozen_cnn":
        from so101.model.model import multitask_cnn_from_checkpoint

        if bundle.cnn_backbone_path is None:
            raise ValueError(
                "encoder_type='frozen_cnn' but cnn_backbone_path is None. "
                "Check the bundle manifest."
            )

        ve_cfg = bundle.manifest["vision_encoder"]
        backbone_cfg = ve_cfg.get("backbone")
        if backbone_cfg is None:
            raise ValueError(
                "manifest.json vision_encoder.backbone is required for frozen_cnn "
                "but was not found."
            )

        # Build a backbone config namespace from the manifest dict
        cnn_model = multitask_cnn_from_checkpoint(
            path=str(bundle.cnn_backbone_path),
            backbone_cfg=backbone_cfg,
            device=device,
        )
        return CnnSpatialSoftmaxFeatureExtractor(
            model=cnn_model,
            device=device,
            target_size=(bundle.image_height, bundle.image_width),
        )

    raise ValueError(
        f"Unknown vision_encoder.type: {encoder_type!r}.\n"
        "Supported types: 'frozen_resnet18', 'frozen_cnn'."
    )

"""image_pipeline.py — Build the deploy-time image preprocessing pipeline from bundle.

Reads ``deploy_image_pipeline.yaml`` and instantiates the corresponding
so101.utils.image_processing steps in order.  All DR augmentations are
absent from this pipeline — the YAML was written by export_bundle.py to
contain only the inference-time steps.
"""

from __future__ import annotations

import yaml

from so101.utils.image_processing.image_pipeline import (
    ImagePipeline,
    Uint8ToFloatCHWPipelineStep,
)

from .bundle import DeployBundle

# ── Step registry ─────────────────────────────────────────────────────────────
# Maps the string ``type`` in the YAML to a factory function.
# Add new step types here as needed.


def _make_uint8_to_float_chw(params: dict) -> Uint8ToFloatCHWPipelineStep:
    return Uint8ToFloatCHWPipelineStep()


_STEP_REGISTRY: dict[str, callable] = {
    "Uint8ToFloatCHW": _make_uint8_to_float_chw,
}


def build_deploy_pipeline(bundle: DeployBundle) -> ImagePipeline:
    """Instantiate the deploy-time image pipeline from the bundle's YAML.

    Parameters
    ----------
    bundle:
        Validated DeployBundle.

    Returns
    -------
    ImagePipeline
        Sequential preprocessing pipeline ready for ``.process(images)``.
    """
    with open(bundle.image_pipeline_path, "r") as f:
        spec = yaml.safe_load(f)

    if not spec or "steps" not in spec:
        raise ValueError(
            f"deploy_image_pipeline.yaml is malformed (missing 'steps' key): "
            f"{bundle.image_pipeline_path}"
        )

    steps = []
    for entry in spec["steps"]:
        step_type = entry.get("type")
        if step_type is None:
            raise ValueError(f"Pipeline step entry is missing 'type': {entry}")
        if step_type not in _STEP_REGISTRY:
            raise ValueError(
                f"Unknown image pipeline step type: {step_type!r}.\n"
                f"Supported types: {sorted(_STEP_REGISTRY)}.\n"
                f"Add a factory for this step in so101_real/image_pipeline.py."
            )
        params = {k: v for k, v in entry.items() if k != "type"}
        steps.append(_STEP_REGISTRY[step_type](params))

    return ImagePipeline(steps)

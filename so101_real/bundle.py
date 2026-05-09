"""bundle.py — Load and validate a deploy bundle produced by export_bundle.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Required manifest fields ──────────────────────────────────────────────────
_REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "task",
    "vision_encoder",
    "policy",
    "deploy_image_pipeline_file",
    "joint_config_file",
    "active_joints",
    "joint_lower_rad",
    "joint_upper_rad",
    "control_hz",
    "actor_obs_metrics",
}


@dataclass(frozen=True)
class DeployBundle:
    """Immutable, validated reference to all files in a deploy bundle."""

    bundle_dir: Path
    manifest: dict[str, Any]

    # ── Derived convenience properties ────────────────────────────────────────

    @property
    def policy_path(self) -> Path:
        return self.bundle_dir / self.manifest["policy"]["file"]

    @property
    def cnn_backbone_path(self) -> Path | None:
        f = self.manifest.get("cnn_backbone_file")
        return (self.bundle_dir / f) if f else None

    @property
    def image_pipeline_path(self) -> Path:
        return self.bundle_dir / self.manifest["deploy_image_pipeline_file"]

    @property
    def joint_config_path(self) -> Path:
        return self.bundle_dir / self.manifest["joint_config_file"]

    @property
    def encoder_type(self) -> str:
        return self.manifest["vision_encoder"]["type"]

    @property
    def image_height(self) -> int:
        return int(self.manifest["vision_encoder"]["image_height"])

    @property
    def image_width(self) -> int:
        return int(self.manifest["vision_encoder"]["image_width"])

    @property
    def obs_dim(self) -> int:
        return int(self.manifest["policy"]["obs_dim"])

    @property
    def act_dim(self) -> int:
        return int(self.manifest["policy"]["act_dim"])

    @property
    def hidden_dims(self) -> list[int]:
        return list(self.manifest["policy"]["hidden_dims"])

    @property
    def active_joints(self) -> list[str]:
        return list(self.manifest["active_joints"])

    @property
    def joint_lower_rad(self) -> list[float]:
        return list(self.manifest["joint_lower_rad"])

    @property
    def joint_upper_rad(self) -> list[float]:
        return list(self.manifest["joint_upper_rad"])

    @property
    def control_hz(self) -> float:
        return float(self.manifest["control_hz"])

    @property
    def actor_obs_metrics(self) -> list[str]:
        return list(self.manifest.get("actor_obs_metrics") or [])


def load_bundle(bundle_dir: str | Path) -> DeployBundle:
    """Load and validate a deploy bundle directory.

    Raises
    ------
    FileNotFoundError
        If the bundle directory or any required file is missing.
    ValueError
        If manifest.json is missing required fields or contains invalid values.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json not found in bundle: {bundle_dir}\n"
            "Re-run `python scripts/run.py export` to regenerate the bundle."
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # ── Schema validation ─────────────────────────────────────────────────────
    missing = _REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise ValueError(
            f"manifest.json is missing required keys: {sorted(missing)}\n"
            f"Bundle path: {bundle_dir}"
        )

    # Verify all referenced files exist
    bundle = DeployBundle(bundle_dir=bundle_dir, manifest=manifest)
    _verify_files(bundle)

    return bundle


def _verify_files(bundle: DeployBundle) -> None:
    """Raise FileNotFoundError if any required bundle file is absent."""
    required = [
        bundle.policy_path,
        bundle.image_pipeline_path,
        bundle.joint_config_path,
    ]
    if bundle.encoder_type == "frozen_cnn":
        if bundle.cnn_backbone_path is None:
            raise ValueError(
                "manifest.json specifies encoder_type='frozen_cnn' but "
                "cnn_backbone_file is not set."
            )
        required.append(bundle.cnn_backbone_path)

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Bundle file missing: {path}\n"
                f"Bundle dir: {bundle.bundle_dir}\n"
                "Re-export the bundle to regenerate missing files."
            )

"""Typed configuration dataclasses mirroring the SO-101 environment YAML schema.

Usage (1-liner in env_cfg.py):
    _Y = So101EnvParams.load(os.environ["SO101_ENV_CONFIG"])
"""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_type_hints

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _optional_dataclass_type(ft: object) -> type | None:
    """Return the inner dataclass type if *ft* is ``Optional[SomeDataclass]``.

    Handles both ``X | None`` (Python 3.10+ ``types.UnionType``) and
    ``typing.Optional[X]`` / ``typing.Union[X, None]``.
    """
    if isinstance(ft, types.UnionType):
        args = ft.__args__
    elif getattr(ft, "__origin__", None) is typing.Union:
        args = ft.__args__
    else:
        return None
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and dataclasses.is_dataclass(non_none[0]):
        return non_none[0]
    return None


def _from_dict(cls: type, data: dict) -> object:
    """Recursively construct a dataclass from a mapping.

    Raises KeyError for unknown keys or missing *required* keys (fields with
    no default value or default factory).  Fields that carry a default are
    silently omitted from ``data`` and receive their default value instead.
    """
    if not dataclasses.is_dataclass(cls):
        return data

    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")

    # Only flag fields that have no default as missing.
    required = {
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"Missing required keys in {cls.__name__}: {sorted(missing)}")

    kwargs: dict = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            # Field has a default; let the dataclass constructor use it.
            continue
        val = data[f.name]
        ft = hints[f.name]
        inner_dc = _optional_dataclass_type(ft)
        if dataclasses.is_dataclass(ft):
            assert isinstance(ft, type)
            kwargs[f.name] = _from_dict(ft, val)
        elif inner_dc is not None:
            # Optional[SomeDataclass] — recurse only when val is not None.
            kwargs[f.name] = None if val is None else _from_dict(inner_dc, val)
        elif getattr(ft, "__origin__", None) is tuple and isinstance(val, list):
            kwargs[f.name] = tuple(val)
        elif getattr(ft, "__origin__", None) is list and isinstance(val, list):
            inner_args = getattr(ft, "__args__", None)
            if (
                inner_args
                and len(inner_args) == 1
                and dataclasses.is_dataclass(inner_args[0])
            ):
                kwargs[f.name] = [_from_dict(inner_args[0], item) for item in val]
            else:
                kwargs[f.name] = val
        else:
            kwargs[f.name] = val

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Sim / scene
# ---------------------------------------------------------------------------


@dataclass
class SimCfg:
    dt: float


@dataclass
class SceneCfg:
    num_envs: int
    env_spacing: float
    replicate_physics: bool


# ---------------------------------------------------------------------------
# Joints / control / safety
# ---------------------------------------------------------------------------


@dataclass
@dataclass
class StartingPositionNoiseCfg:
    enabled: bool
    range: tuple[float, float]


@dataclass
class JointsCfg:
    all: list[str]
    active: list[str]
    wrist_roll_name: str
    starting_position: list
    starting_position_noise: StartingPositionNoiseCfg


@dataclass
class ControlCfg:
    action_scale: float


@dataclass
class SafetyCfg:
    max_joint_velocity: float
    min_ee_height: float


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------


@dataclass
class GripperCfg:
    ee_link_name: str
    grip_zone_rot: tuple[float, float, float, float]
    tip_offset: tuple[float, float, float]
    open_target: float
    closed_target: float


# ---------------------------------------------------------------------------
# Distractors
# ---------------------------------------------------------------------------


@dataclass
class DistractorGeometryCfg:
    cube_size: tuple[float, float, float]
    sphere_radius: float
    cone_radius: float
    cone_height: float


@dataclass
class DistractorRandomizationCfg:
    enabled: bool
    size_randomization_enabled: bool
    active_probability: float
    size_range: tuple[float, float]


@dataclass
class DistractorPositionCfg:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]


@dataclass
class DistractorsCfg:
    count: int
    geometry: DistractorGeometryCfg
    randomization: DistractorRandomizationCfg
    position: DistractorPositionCfg


# ---------------------------------------------------------------------------
# Debug / behavior
# ---------------------------------------------------------------------------


@dataclass
class SaveImageStageCfg:
    save: bool
    interval: int


@dataclass
class SaveImagesCfg:
    pre_processing: SaveImageStageCfg
    post_processing: SaveImageStageCfg


@dataclass
class VisionDebugItemCfg:
    enabled: bool


@dataclass
class ConvLayerMapsCfg:
    enabled: bool
    max_channels: int  # max feature channels to tile per layer


@dataclass
class VisionDebugCfg:
    enabled: bool
    interval: int  # log every N environment steps
    num_envs_logged: int  # number of envs visualised (taken from env 0..N)
    raw_image: VisionDebugItemCfg
    pipelined_image: VisionDebugItemCfg
    conv_layer_maps: ConvLayerMapsCfg
    activation_heatmap: VisionDebugItemCfg
    keypoints: VisionDebugItemCfg


@dataclass
class DebugCfg:
    save_images: SaveImagesCfg
    enable_gripper_arrow_markers: bool
    enable_camera_frame_markers: bool
    enable_grip_zone_markers: bool
    vision_debug: VisionDebugCfg


@dataclass
class BinaryGripperActionCfg:
    enabled: bool


@dataclass
class BehaviorCfg:
    binary_gripper_action: BinaryGripperActionCfg


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@dataclass
class GateCfg:
    """A single gate condition: reward fires only when ``metric op threshold``.

    Exactly one of ``gt`` / ``gte`` / ``lt`` / ``lte`` / ``eq`` must be set.
    Multiple :class:`GateCfg` on a reward step are conjunctive (ALL must pass).
    """

    metric: str
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None
    eq: float | None = None

    def __post_init__(self) -> None:
        ops = [
            x for x in (self.gt, self.gte, self.lt, self.lte, self.eq) if x is not None
        ]
        if len(ops) != 1:
            raise ValueError(
                f"GateCfg for metric '{self.metric}' must declare exactly one "
                f"operator (gt/gte/lt/lte/eq); got {len(ops)}."
            )


@dataclass
class RewardCfg:
    enabled: bool
    scale: float
    mode: str = "absolute"
    """Reward computation mode: 'absolute' (current value) or 'progressive'
    (improvement delta from previous step, clamped to min=0)."""
    gates: list[GateCfg] = field(default_factory=list)
    """Optional gate conditions. Reward (and termination for terminal steps) is
    suppressed for any environment where a gate condition is not met."""


@dataclass(kw_only=True)
class DistanceRewardCfg(RewardCfg):
    distance_pressure: float


@dataclass(kw_only=True)
class GripCubeRewardCfg(RewardCfg):
    distance_threshold: float
    touch_force_threshold: float


@dataclass(kw_only=True)
class CloseGripperRewardCfg(RewardCfg):
    close_target: float
    max_open: float


@dataclass(kw_only=True)
class EeLinearSpeedRewardCfg(RewardCfg):
    safe_speed: float


@dataclass(kw_only=True)
class SuccessLiftFractionRewardCfg(RewardCfg):
    height_threshold: float


@dataclass
class ApproachDistanceMetricCfg:
    pressure: float
    distance_max: float
    linear_weight: float


@dataclass
class ApproachAlignmentMetricCfg:
    pressure: float
    linear_weight: float


@dataclass
class ApproachGripperPoseMetricCfg:
    gripper_pos_target: float
    pressure: float
    linear_weight: float


@dataclass
class ApproachPhaseMetricCfg:
    distance_pressure: float
    alignment_pressure: float
    gripper_pos_pressure: float
    gripper_pos_target: float


@dataclass
class MetricsCfg:
    approach_distance: ApproachDistanceMetricCfg
    approach_alignment: ApproachAlignmentMetricCfg
    approach_gripper_pose: ApproachGripperPoseMetricCfg
    approach_phase: ApproachPhaseMetricCfg


@dataclass(kw_only=True)
class GraspPhaseRewardCfg(RewardCfg):
    grip_force_pressure: float
    grip_force_target: float


@dataclass(kw_only=True)
class AvoidBumpingCubeRewardCfg(RewardCfg):
    cube_widths: float  # multiplier on CUBE_WIDTH to define "near cube" region


@dataclass(kw_only=True)
class CubeOutOfRangeTerminalRewardCfg(RewardCfg):
    distance_threshold: (
        float  # metres; cube further than this from robot base triggers termination
    )


@dataclass(kw_only=True)
class ApproachPhaseTerminalRewardCfg(RewardCfg):
    threshold: float


@dataclass(kw_only=True)
class GraspPhaseTerminalRewardCfg(RewardCfg):
    threshold: float


@dataclass(kw_only=True)
class WristRollPoseRewardCfg(RewardCfg):
    target_rad: float
    """Target wrist roll joint position in radians. -1.5707963267948966 = -90°."""
    pressure: float
    """Exponential pressure: reward = exp(-pressure * |q - target_rad|) * scale."""


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------


@dataclass
class PreshapeImageCfg:
    enabled: bool


@dataclass
class GaussianNoiseCfg:
    enabled: bool
    std_range: list[float]


@dataclass
class BrightnessCfg:
    enabled: bool
    range: list[float]


@dataclass
class ContrastCfg:
    enabled: bool
    range: list[float]


@dataclass
class CameraFeedCfg:
    preshape_image: PreshapeImageCfg
    gaussian_noise: GaussianNoiseCfg
    brightness: BrightnessCfg
    contrast: ContrastCfg
    gaussian_blur: GaussianBlurCfg
    cheap_webcam_effect: CheapWebcamEffectCfg
    motion_blur: MotionBlurCfg
    jpeg_compression: JpegCompressionCfg


@dataclass
class GaussianBlurCfg:
    enabled: bool
    kernel_size: int
    sigma: float


@dataclass
class CheapWebcamEffectCfg:
    enabled: bool


@dataclass
class MotionBlurCfg:
    enabled: bool
    kernel_size: int
    strength_range: list[float]


@dataclass
class JpegCompressionCfg:
    enabled: bool
    quality_range: list[int]


@dataclass
class CameraPoseCfg:
    enabled: bool
    position_noise_range: tuple[float, float]
    rotation_noise_deg_range: tuple[float, float]


@dataclass
class DRCameraCfg:
    feed: CameraFeedCfg
    pose: CameraPoseCfg


@dataclass
class WorldLightingCfg:
    enabled: bool
    intensity_range: tuple[float, float]
    color_variation: float


@dataclass
class EnvLightingCfg:
    enabled: bool
    height_range: list[float]
    intensity_range: list[float]
    color_variation: float
    specular_range: list[float]


@dataclass
class CubePositionRandomizationCfg:
    enabled: bool
    radius_range: tuple[float, float]
    angle_range: tuple[float, float]
    z_range: tuple[float, float]


@dataclass
class DRCubeCfg:
    color_randomization_enabled: bool
    size_randomization_enabled: bool
    size_range: tuple[float, float]
    position_randomization: CubePositionRandomizationCfg


@dataclass
class GroundDRCfg:
    enabled: bool


@dataclass
class DomainRandomizationCfg:
    camera: DRCameraCfg
    world_lighting: WorldLightingCfg
    env_lighting: EnvLightingCfg
    cube: DRCubeCfg
    ground: GroundDRCfg


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


@dataclass
class CameraSensorCfg:
    height: int
    width: int


@dataclass
class DebugVisCfg:
    debug_vis: bool


@dataclass
class TableContactSensorCfg:
    track_pose: bool
    debug_vis: bool


@dataclass
class SensorsCfg:
    camera: CameraSensorCfg
    gripper_contact: DebugVisCfg
    table_contact: TableContactSensorCfg
    gripper_transform: DebugVisCfg


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass
class ObservationsCfg:
    critic_obs_metrics: list[str]


@dataclass
class CnnBackboneCfg:
    """Architecture for the CNN backbone inside :class:`~so101.model.model.MultiTaskCnn`.

    Used by the ``frozen_cnn`` vision encoder type.  Must be set in the env
    config YAML under ``vision_encoder.backbone``.
    """

    channels: list
    kernel_sizes: list
    strides: list
    mlp_hidden_dims: list
    output_dim: int


@dataclass
class VisionEncoderCfg:
    """Configuration for the vision feature extractor used in the actor policy.

    type:
        'frozen_resnet18' — frozen pretrained ResNet18 + SpatialSoftmax (1024-D output).
        'frozen_cnn'      — frozen pretrained lightweight CNN + SpatialSoftmax.
                            Architecture is defined in ``backbone``.  Backbone weights
                            are loaded from ``cnn_checkpoint`` at env construction.
    """

    type: str
    image_height: int
    image_width: int
    backbone: CnnBackboneCfg | None = None
    cnn_checkpoint: str | None = None


@dataclass
class So101EnvParams:
    decimation: int
    episode_length_s: float
    sim: SimCfg
    scene: SceneCfg
    joints: JointsCfg
    control: ControlCfg
    safety: SafetyCfg
    gripper: GripperCfg
    distractors: DistractorsCfg
    debug: DebugCfg
    behavior: BehaviorCfg
    metrics: MetricsCfg
    rewards: list[dict]
    domain_randomization: DomainRandomizationCfg
    sensors: SensorsCfg
    observations: ObservationsCfg
    vision_encoder: VisionEncoderCfg

    def get_reward_cfg(self, type_name: str) -> types.SimpleNamespace:
        """Return the first reward list entry with matching ``type`` as a namespace.

        Used by metric steps to access reward-type parameters (e.g. thresholds)
        without requiring the reward step to be enabled.
        Raises ``KeyError`` if no entry with the given type is found.
        """
        for entry in self.rewards:
            if entry.get("type") == type_name:
                return types.SimpleNamespace(
                    **{k: v for k, v in entry.items() if k != "type"}
                )
        raise KeyError(
            f"No reward list entry with type='{type_name}'. "
            f"Present types: {[e.get('type') for e in self.rewards]}"
        )

    @classmethod
    def load(cls, path: str | Path) -> "So101EnvParams":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"SO101 env config not found at '{config_path}'.")
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError("Top-level YAML must be a mapping.")
        print(f"[So101EnvParams] Loading from: {config_path}")
        return _from_dict(cls, data)  # type: ignore[return-value]

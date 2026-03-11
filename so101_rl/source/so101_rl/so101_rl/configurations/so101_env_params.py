"""Typed configuration dataclasses mirroring the SO-101 environment YAML schema.

Usage (1-liner in env_cfg.py):
    _Y = So101EnvParams.load(os.environ["SO101_ENV_CONFIG"])
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _from_dict(cls: type, data: dict) -> object:
    """Recursively construct a dataclass from a mapping.

    Raises KeyError for unknown or missing keys at every level.
    """
    if not dataclasses.is_dataclass(cls):
        return data

    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")

    missing = known - set(data)
    if missing:
        raise KeyError(f"Missing required keys in {cls.__name__}: {sorted(missing)}")

    kwargs: dict = {}
    for f in dataclasses.fields(cls):
        val = data[f.name]
        ft = hints[f.name]
        if dataclasses.is_dataclass(ft):
            assert isinstance(ft, type)
            kwargs[f.name] = _from_dict(ft, val)
        elif getattr(ft, "__origin__", None) is tuple and isinstance(val, list):
            kwargs[f.name] = tuple(val)
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
    grip_zone_offset: tuple[float, float, float]
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
class DebugCfg:
    save_images: bool
    save_image_interval: int
    enable_gripper_arrow_markers: bool
    enable_tip_markers: bool
    enable_camera_frame_markers: bool


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
class RewardCfg:
    enabled: bool
    scale: float


@dataclass
class GripCubeRewardCfg:
    enabled: bool
    scale: float
    distance_threshold: float


@dataclass
class CloseGripperRewardCfg:
    enabled: bool
    scale: float
    close_target: float
    max_open: float


@dataclass
class GripperForceRewardCfg:
    enabled: bool
    scale: float
    force_target: float


@dataclass
class EeLinearSpeedRewardCfg:
    enabled: bool
    scale: float
    safe_speed: float


@dataclass
class SuccessLiftFractionRewardCfg:
    enabled: bool
    scale: float
    height_threshold: float


@dataclass
class SuccessTouchTerminalRewardCfg:
    enabled: bool
    scale: float
    touch_force_threshold: float


@dataclass
class VantageRewardCfg:
    enabled: bool
    scale: float
    ideal_distance: float
    ideal_distance_sigma: float
    ideal_height: float
    ideal_height_sigma: float
    min_distance_threshold: float
    far_distance_threshold: float


@dataclass
class RewardsCfg:
    distance: RewardCfg
    grip_cube: GripCubeRewardCfg
    lift_cube: RewardCfg
    gripper_cube_alignment: RewardCfg
    camera_cube_alignment: RewardCfg
    close_gripper: CloseGripperRewardCfg
    gripper_force: GripperForceRewardCfg
    gripper_look_at_cube: RewardCfg
    action: RewardCfg
    ee_linear_speed: EeLinearSpeedRewardCfg
    joint_speed: RewardCfg
    ee_height_safety: RewardCfg
    safety_touch_table: RewardCfg
    success_lift_fraction_terminal: SuccessLiftFractionRewardCfg
    success_touch_terminal: SuccessTouchTerminalRewardCfg
    success_point_at_cube_terminal: RewardCfg
    safety_touch_table_terminal: RewardCfg
    vantage: VantageRewardCfg
    keep_camera_upright: RewardCfg


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------

@dataclass
class PreshapeImageCfg:
    enabled: bool


@dataclass
class ContrastCfg:
    enabled: bool
    range: list[float]


@dataclass
class CameraFeedCfg:
    preshape_image: PreshapeImageCfg
    gaussian_noise_std_range: list[float]
    brightness_range: list[float]
    contrast: ContrastCfg


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
class CameraAdvancedCfg:
    motion_blur: MotionBlurCfg
    jpeg_compression: JpegCompressionCfg


@dataclass
class CameraPoseCfg:
    enabled: bool
    position_noise_range: tuple[float, float]
    rotation_noise_deg_range: tuple[float, float]


@dataclass
class DRCameraCfg:
    feed: CameraFeedCfg
    advanced: CameraAdvancedCfg
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
    grip_zone_transform: DebugVisCfg


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@dataclass
class ObservationsCfg:
    critic_obs_metrics: list[str]


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
    rewards: RewardsCfg
    domain_randomization: DomainRandomizationCfg
    sensors: SensorsCfg
    observations: ObservationsCfg

    @classmethod
    def load(cls, path: str | Path) -> "So101EnvParams":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"SO101 env config not found at '{config_path}'."
            )
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError("Top-level YAML must be a mapping.")
        print(f"[So101EnvParams] Loading from: {config_path}")
        return _from_dict(cls, data)  # type: ignore[return-value]

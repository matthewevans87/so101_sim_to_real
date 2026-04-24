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
    enable_goal_zone_markers: bool
    vision_debug: VisionDebugCfg


@dataclass
class BinaryGripperActionCfg:
    enabled: bool


@dataclass
class BehaviorCfg:
    binary_gripper_action: BinaryGripperActionCfg
    max_cube_distance: float


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
    """Reward computation mode.

    ``'absolute'``: reward = base * scale each step.
    ``'unsigned_progressive'``: reward = max(0, \u0394base) * scale; only improvements
    are rewarded, regressions yield zero.
    ``'signed_progressive'``: reward = \u0394base * scale; regressions yield negative
    reward, discouraging lift\u2192lower\u2192lift cycles."""
    gates: list[GateCfg] = field(default_factory=list)
    """Optional gate conditions. Reward is suppressed for any environment where
    a gate condition is not met."""
    fire_once: bool = False
    """When ``True``, the reward fires at most once per episode, on the first
    step the (post-gate) reward is non-zero.  Subsequent steps yield no
    additional reward.  Valid for any reward step type.  Use with ``gates`` to
    express milestone bonuses (e.g. gate on ``cube_height_w >= 0.10``)."""
    id: str | None = None
    """Optional human-readable identifier for this reward instance.

    When set, the TensorBoard logging key becomes ``Episode_Reward/<type>[<id>]``
    instead of ``Episode_Reward/<type>``.  This is required when the same
    ``type`` appears more than once in the rewards list (e.g. two
    ``approach_phase`` entries with different modes) so that each instance
    can be monitored and targeted by sweep overrides independently.

    Sweep override entries are matched by ``(type, id)`` when ``id`` is
    present, or by ``type`` alone when ``id`` is absent."""


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@dataclass
class TerminationCfg:
    """Configuration for a single :class:`TerminationCondition`.

    A termination condition fires when its gate predicate evaluates True for
    an environment.  Firing causes the episode to end (terminated, not
    time-out).  The :class:`TerminationPipeline` ORs together the enabled
    conditions to produce the final terminal mask.

    There is intentionally no ``terminate`` field: presence in the
    ``terminations`` list IS the termination intent.  Reward shaping that
    should accompany termination is configured separately in ``rewards``
    (typically with ``fire_once: true`` and matching gates).
    """

    enabled: bool
    is_success: bool
    """When True, this condition is treated as the task-success terminal by
    the eval pipeline (used to classify the primary cause of an episode end).
    Exactly one enabled termination condition must declare ``is_success: true``;
    cross-field validation enforces this."""
    gates: list[GateCfg] = field(default_factory=list)
    """Gate conditions; the condition fires only when ALL gates pass.  An empty
    gates list means the condition fires every step for every environment
    (i.e. an immediate termination), which is almost never what is wanted —
    cross-field validation rejects empty gates on enabled conditions."""
    id: str | None = None
    """Required when more than one TerminationCfg of the same registered type
    is present.  Also serves as the TensorBoard logging key suffix
    (``Termination/<id>``)."""


@dataclass(kw_only=True)
class DistanceRewardCfg(RewardCfg):
    distance_pressure: float


@dataclass(kw_only=True)
class CloseGripperRewardCfg(RewardCfg):
    close_target: float
    max_open: float


@dataclass(kw_only=True)
class EeLinearSpeedRewardCfg(RewardCfg):
    safe_speed: float


@dataclass
class ApproachDistanceMetricCfg:
    pressure: float
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
class GraspPhaseMetricCfg:
    grip_force_sat_threshold: float
    """Bilateral pinch force (N) at which grasp_phase saturates to 1.0."""


@dataclass
class GripCubeMetricCfg:
    distance_threshold: float
    """Grip-zone-to-cube distance (m) within which the cube is in grip position."""
    touch_force_threshold: float
    """Minimum gripper contact force (N) to count as a confirmed grip."""


@dataclass
class CubeOutOfRangeMetricCfg:
    distance_threshold: float
    """Cube-to-base distance (m) beyond which the cube is considered out of range."""


@dataclass
class LiftPhaseMetricCfg:
    height_threshold: float
    """Cube height above resting position (m) at which cube_lift_fraction reaches 1.0."""


@dataclass
class ApproachPhaseTerminalMetricCfg:
    threshold: float
    """approach_phase value above which approach_phase_terminal is True."""


@dataclass
class GraspPhaseTerminalMetricCfg:
    threshold: float
    """grasp_phase value above which grasp_phase_terminal is True (when approach is also terminal)."""


@dataclass
class GoalZoneDistanceMetricCfg:
    """Configuration for the goal zone distance metric.

    ``goal_zone_distance`` is a shaped scalar in (0, 1] that is highest (1.0)
    when the cube centroid coincides with the goal zone and decays
    exponentially with distance.
    """

    pressure: float
    """Exponential decay rate: ``exp(-pressure * dist / env_cube_width)``.
    Higher values create a sharper peak around the goal zone."""
    distance_threshold: float
    """Euclidean distance (m) within which the cube centroid is considered to
    have *reached* the goal zone (``is_goal_zone_reached = True``)."""


@dataclass
class MetricsCfg:
    approach_distance: ApproachDistanceMetricCfg
    approach_alignment: ApproachAlignmentMetricCfg
    approach_gripper_pose: ApproachGripperPoseMetricCfg
    approach_phase: ApproachPhaseMetricCfg
    grasp_phase: GraspPhaseMetricCfg
    grip_cube: GripCubeMetricCfg
    cube_out_of_range: CubeOutOfRangeMetricCfg
    lift_phase: LiftPhaseMetricCfg
    approach_phase_terminal: ApproachPhaseTerminalMetricCfg
    grasp_phase_terminal: GraspPhaseTerminalMetricCfg
    goal_zone_distance: GoalZoneDistanceMetricCfg


@dataclass(kw_only=True)
class AvoidBumpingCubeRewardCfg(RewardCfg):
    cube_widths: float  # multiplier on CUBE_WIDTH to define "near cube" region


@dataclass(kw_only=True)
class WristRollPoseRewardCfg(RewardCfg):
    target_rad: float
    """Target wrist roll joint position in radians. -1.5707963267948966 = -90°."""
    pressure: float
    """Exponential pressure: reward = exp(-pressure * |q - target_rad|) * scale."""


@dataclass(kw_only=True)
class ActionRewardCfg(RewardCfg):
    joints: list[str] = field(default_factory=list)
    """Joint names to include in the action-smoothing penalty.
    An empty list means all active joints are used."""


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
class GoalZonePositionDRCfg:
    """Domain randomisation config for the per-episode goal zone target position.

    The goal zone is sampled in polar coordinates relative to the robot base
    at episode reset.  Set ``enabled: false`` to disable the feature entirely
    (no :class:`GoalZoneEnvMetricStep` will be added to the pipeline).
    """

    enabled: bool
    radius_range: tuple[float, float]
    """Radial distance from the robot base (m) in which the goal zone is sampled."""
    angle_range: tuple[float, float]
    """Azimuthal angle range (degrees) for the goal zone."""
    z_range: tuple[float, float]
    """Height range (m) above the table surface for the goal zone."""


@dataclass
class DomainRandomizationCfg:
    camera: DRCameraCfg
    world_lighting: WorldLightingCfg
    env_lighting: EnvLightingCfg
    cube: DRCubeCfg
    ground: GroundDRCfg
    goal_zone: GoalZonePositionDRCfg


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
    moving_jaw_contact: DebugVisCfg
    table_contact: TableContactSensorCfg
    gripper_transform: DebugVisCfg


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass
class ObservationsCfg:
    critic_obs_metrics: list[str] = field(default_factory=list)
    """Extra metric keys appended to the critic observation vector.

    NOTE: As of skrl 1.4.3 the installed ``IsaacLabWrapper.step`` discards
    ``observations["critic"]`` and feeds only ``observations["policy"]`` to
    both the policy and value networks.  Until skrl is upgraded to a
    wrapper that routes the critic dict (see ``develop`` branch), entries
    here are *computed but never seen by the critic network*.  Default is
    an empty list so configs can omit the key entirely.
    """
    actor_obs_metrics: list[str] = field(default_factory=list)
    """Extra metric keys appended to the actor observation vector after the
    frozen vision features and joint positions.  Defaults to an empty list
    (no change to actor obs space).  Each key must appear in
    :data:`KEY_OBS_DIMS` so its column width is known."""
    telemetry_metrics: list[str] = field(default_factory=list)
    """Extra metric keys to include in the metric pipeline for telemetry collection.
    Unlike critic_obs_metrics, these do not become part of the critic observation
    vector — they are computed and exposed via step_metrics but not fed to the model.
    Adding keys here does not change the observation space or invalidate checkpoints."""
    critic_include_vision_features: bool = False
    """When True, prepend the same frozen vision features used by the actor into the
    critic observation vector.  Default False preserves the privileged-state-only
    critic and keeps backward compatibility with saved checkpoints."""


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
class EpisodeStatsCfg:
    lift_height_threshold: float
    """Cube height above resting position (m) at which the cube is considered
    'lifted' for the purposes of ``lift_rate``, ``drop_rate``, and
    ``time_to_lift`` statistics.  Should be a small positive value (e.g. 0.01)
    to detect any departure from the table, rather than the full
    ``lift_phase.height_threshold`` used for the success condition."""


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
    episode_stats: EpisodeStatsCfg
    rewards: list[dict]
    terminations: list[dict]
    domain_randomization: DomainRandomizationCfg
    sensors: SensorsCfg
    observations: ObservationsCfg
    vision_encoder: VisionEncoderCfg

    def get_reward_cfg(self, type_name: str) -> types.SimpleNamespace:
        """Return the first reward list entry with matching ``type`` as a namespace.

        Used by metric steps to access reward-type parameters (e.g. thresholds)
        without requiring the reward step to be enabled.
        Raises ``KeyError`` if no entry with the given type is found.
        Supports both the new list format (``[{type: ..., ...}]``) and the old
        named-map format (``{type_name: {...}}``) for backward compatibility with
        saved experiment configs.
        """
        if isinstance(self.rewards, dict):
            # Old named-map format.
            if type_name in self.rewards:
                return types.SimpleNamespace(**self.rewards[type_name])
            raise KeyError(
                f"No reward entry with type='{type_name}'. "
                f"Present types: {sorted(self.rewards)}"
            )
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
        cls._migrate_legacy_termination_schema(data, config_path)
        params = _from_dict(cls, data)  # type: ignore[assignment]
        params._validate_cross_field_invariants(config_path)
        return params

    @staticmethod
    def _migrate_legacy_termination_schema(data: dict, config_path: Path) -> None:
        """Auto-migrate pre-Phase-A configs (``terminate`` field on rewards)
        to the new ``terminations`` list schema, in-place on *data*.

        Detection: a config is considered legacy when no top-level
        ``terminations`` key is present.  Migration generates termination
        entries from any reward whose ``terminate`` field is True (the old
        default for terminal reward steps), then strips the ``terminate``
        field from every reward entry so :class:`RewardCfg` (which no longer
        carries ``terminate``) accepts it.

        This shim exists so frozen ``env_config.yaml`` files in past sweep
        dirs can be re-evaluated under the new pipeline without retraining.
        """
        if "terminations" in data:
            return

        rewards = data.get("rewards")
        if rewards is None:
            raise KeyError(
                f"{config_path}: missing required top-level keys 'rewards' "
                f"and 'terminations'."
            )
        if isinstance(rewards, dict):
            # Pre-list named-map format: normalise to list first.
            rewards = [{"type": k, **v} for k, v in rewards.items()]
            data["rewards"] = rewards

        # Map of legacy TerminalRewardStep types to
        # (gate_metric, threshold_op, threshold_value, is_success).
        # ``static`` is special: it carries its own gates, so we use the
        # entry's gates verbatim (spec value None).
        legacy_terminal_metrics: dict[str, tuple[str, str, float, bool] | None] = {
            "success_lift_fraction_terminal": (
                "is_success_lift_fraction_terminal",
                "gte",
                0.5,
                True,
            ),
            "safety_touch_table_terminal": (
                "is_table_touched",
                "gte",
                0.5,
                False,
            ),
            "cube_out_of_range_terminal": (
                "is_cube_out_of_range",
                "gte",
                0.5,
                False,
            ),
            "approach_phase_terminal": (
                "approach_phase_terminal",
                "gte",
                0.5,
                False,
            ),
            "grasp_phase_terminal": (
                "grasp_phase_terminal",
                "gte",
                0.5,
                False,
            ),
            "static": None,
        }

        terminations: list[dict] = []
        success_seen = False
        for entry in rewards:
            type_name = entry.get("type")
            # OLD schema default for ``terminate`` was True (the field lived on
            # RewardCfg with default True).  Legacy *terminal* reward types
            # therefore inherit terminate=True when the field is absent;
            # all other reward types have always required terminate=False
            # implicitly (they are not :class:`TerminalRewardStep` subclasses).
            legacy_default_terminate = type_name in legacy_terminal_metrics
            terminate = entry.pop("terminate", legacy_default_terminate)
            if not terminate:
                continue
            if not entry.get("enabled", True):
                continue
            if type_name not in legacy_terminal_metrics:
                raise ValueError(
                    f"{config_path}: legacy reward entry of type "
                    f"'{type_name}' has terminate=True, but no migration "
                    f"rule is defined.  Add an explicit 'terminations:' "
                    f"block to this config."
                )
            spec = legacy_terminal_metrics[type_name]
            term_id = entry.get("id") or type_name
            if spec is None:
                # ``static``: use the entry's gates verbatim.
                gates = list(entry.get("gates", []))
                if not gates:
                    raise ValueError(
                        f"{config_path}: legacy 'static' reward entry "
                        f"id={term_id!r} has terminate=True but no gates; "
                        f"cannot migrate."
                    )
                # In every legacy config in this repo, a ``static`` reward
                # with terminate=True is the lift-success terminal (gates
                # check ``cube_lift_fraction``).  Auto-classify the first
                # such entry as is_success.
                is_success_flag = not success_seen
            else:
                metric, op, value, is_success_flag = spec
                gates = [{"metric": metric, op: value}]
            if is_success_flag:
                success_seen = True
            terminations.append(
                {
                    "enabled": True,
                    "is_success": is_success_flag,
                    "id": term_id,
                    "gates": gates,
                }
            )

        if not terminations:
            raise ValueError(
                f"{config_path}: legacy schema migration produced no "
                f"termination conditions (no rewards had terminate=True). "
                f"Add an explicit 'terminations:' block to this config."
            )
        data["terminations"] = terminations

    def _validate_cross_field_invariants(self, config_path: Path) -> None:
        """Cross-field invariants that single-dataclass validation cannot express.

        The two lift-related thresholds live in different sections (so they
        can be swept independently), but they have a fixed semantic ordering:
        a cube must be 'lifted' (episode_stats.lift_height_threshold) before
        it can be a 'success' (metrics.lift_phase.height_threshold).  A typo
        in either is otherwise silent — caught here loudly.
        """
        lift_min = self.episode_stats.lift_height_threshold
        lift_success = self.metrics.lift_phase.height_threshold
        if not (0.0 < lift_min <= lift_success):
            raise ValueError(
                f"Invalid lift threshold ordering in {config_path}:\n"
                f"  episode_stats.lift_height_threshold     = {lift_min} m  "
                f"(any-lift detection)\n"
                f"  metrics.lift_phase.height_threshold     = {lift_success} m  "
                f"(success threshold)\n"
                f"Required: 0 < episode_stats.lift_height_threshold "
                f"<= metrics.lift_phase.height_threshold."
            )

        # ------------------------------------------------------------------
        # Termination pipeline invariants
        # ------------------------------------------------------------------
        # Each enabled termination must declare at least one gate (an
        # always-firing condition would terminate every step of every episode
        # immediately) and exactly one enabled termination must be flagged
        # ``is_success: true`` so the eval pipeline can classify success
        # episodes without name matching.
        enabled_terms: list[dict] = []
        for entry in self.terminations:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"{config_path}: each entry in 'terminations' must be a mapping; "
                    f"got {type(entry).__name__}."
                )
            if not entry.get("enabled", True):
                continue
            enabled_terms.append(entry)
            term_id = entry.get("id")
            gates = entry.get("gates") or []
            if not gates:
                raise ValueError(
                    f"{config_path}: enabled termination id={term_id!r} has no "
                    f"gates; an empty-gate termination would fire every step. "
                    f"Add explicit gate conditions."
                )
        if not enabled_terms:
            raise ValueError(
                f"{config_path}: 'terminations' list contains no enabled entries. "
                f"At least one termination condition (the success terminal) must be "
                f"enabled, otherwise episodes can only end via time-out."
            )
        success_terms = [e for e in enabled_terms if e.get("is_success", False)]
        if len(success_terms) != 1:
            ids = [e.get("id") for e in success_terms]
            raise ValueError(
                f"{config_path}: exactly one enabled termination must declare "
                f"is_success=true; found {len(success_terms)} (ids={ids}). "
                f"The success terminal is required so eval can classify the "
                f"primary termination cause without name matching."
            )

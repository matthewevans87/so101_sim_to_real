"""controller.py — Real-robot inference loop.

The :class:`InferenceLoop` is **policy-agnostic** and **observation-agnostic**:
it is driven by an injected :class:`~so101_real.policies.Policy` callable and
an :class:`ObsBuilder` that supplies the observation tensor each tick.

Each tick:
  1. Read joint positions (canonical radians) from the robot.
  2. Build the observation via the injected :class:`ObsBuilder`.
  3. Call the policy to obtain a normalized action in ``[-1, 1]``.
  4. Map the action → canonical-radian joint targets using the supplied bounds.
  5. Apply EMA smoothing.
  6. Apply safety clamps (delta + joint limits).
  7. Send the targets to the robot (skipped in dry-run).
  8. Side-effects: optional recorder, overlay, and ROS publisher.

Two observation builders ship in this module:

* :class:`NullObsBuilder` — returns a zero scalar tensor and no camera frame.
  Use with :class:`~so101_real.policies.StaticPositionPolicy`.
* :class:`VisionJointObsBuilder` — grabs a camera frame, runs the deploy
  image pipeline, extracts vision features, and concatenates them with the
  measured joint positions.  Reproduces the trained-policy observation
  contract from sim.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import torch
import yaml

from .bundle import DeployBundle
from .camera import CameraSource
from .image_pipeline import build_deploy_pipeline
from .recorder import EpisodeRecorder
from .robot import RobotConfig, So101Robot, ResetPoseCfg
from .ros_publisher import RosPublisher
from .safety import SafetyLayer
from .units import JointUnitConverter
from .vision import build_vision_encoder

# ── ControllerConfig ──────────────────────────────────────────────────────────


@dataclass
class ControllerConfig:
    """Explicit inference loop configuration.

    Load from YAML via ``ControllerConfig.load(path)``.  All fields are
    required and must be set explicitly in the config file — there are no
    silent defaults.
    """

    ema_alpha: float
    """EMA smoothing coefficient for joint targets. 1.0 = no smoothing, 0.0 = freeze."""

    device: str
    """PyTorch device string for policy and vision encoder inference."""

    control_hz: float
    """Inference loop rate in Hz.  Used by every command (bundle or static)."""

    @classmethod
    def load(cls, path: str | Path) -> "ControllerConfig":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Robot config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        ctrl = data.get("controller")
        if ctrl is None:
            raise ValueError(
                f"Robot config YAML must contain a top-level 'controller' key: {path}"
            )
        required = {"ema_alpha", "device", "control_hz"}
        missing = required - set(ctrl)
        if missing:
            raise ValueError(
                f"Controller config is missing required keys: {sorted(missing)}\n"
                f"Config path: {path}"
            )
        ema_alpha = float(ctrl["ema_alpha"])
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(
                f"controller.ema_alpha must be in (0, 1]; got {ema_alpha}."
            )
        control_hz = float(ctrl["control_hz"])
        if control_hz <= 0.0:
            raise ValueError(f"controller.control_hz must be > 0; got {control_hz}.")
        return cls(
            ema_alpha=ema_alpha,
            device=str(ctrl["device"]),
            control_hz=control_hz,
        )


# ── Observation builders ──────────────────────────────────────────────────────


class ObsBuilder(Protocol):
    """Construct the per-tick observation passed to the policy."""

    def build(self, q_meas: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Return the observation tensor for the current tick.

        Parameters
        ----------
        q_meas:
            Measured joint positions in canonical radians, shape ``(n_joints,)``,
            already on the inference device.

        Returns
        -------
        torch.Tensor
            Observation tensor with a leading batch dimension, shape
            ``(1, obs_dim)``.
        """
        ...

    @property
    def last_frame_rgb(self) -> Optional[np.ndarray]:  # pragma: no cover
        """Most recent camera frame this builder produced, or ``None``.

        Consumed by the recorder and overlay; builders that do not use a
        camera return ``None``.
        """
        ...


class NullObsBuilder:
    """Observation builder for camera-free policies (e.g. static)."""

    def __init__(self, device: torch.device) -> None:
        # A single zero scalar — policy is expected to ignore this entirely.
        self._obs = torch.zeros((1, 1), dtype=torch.float32, device=device)

    def build(self, q_meas: torch.Tensor) -> torch.Tensor:
        return self._obs

    @property
    def last_frame_rgb(self) -> Optional[np.ndarray]:
        return None


class VisionJointObsBuilder:
    """Vision-features + joint-positions observation builder.

    Matches the training observation contract: ``[vision_features | q_meas]``.

    Parameters
    ----------
    bundle:
        Deploy bundle (provides the image pipeline and vision encoder).
    camera:
        Open camera source.
    device:
        PyTorch device for image processing and encoding.
    """

    def __init__(
        self,
        bundle: DeployBundle,
        camera: CameraSource,
        device: torch.device,
    ) -> None:
        if bundle.actor_obs_metrics:
            raise ValueError(
                f"Deploy bundle contains actor_obs_metrics={bundle.actor_obs_metrics} "
                "which are not supported at deploy time.\n"
                "Re-export the bundle after removing these metrics from the env "
                "config, or add support for them in so101_real/controller.py."
            )
        self._camera = camera
        self._pipeline = build_deploy_pipeline(bundle)
        self._encoder = build_vision_encoder(bundle, device=device)
        self._device = device
        self._last_frame_rgb: Optional[np.ndarray] = None

    def build(self, q_meas: torch.Tensor) -> torch.Tensor:
        frame_rgb = self._camera.get_frame_rgb()  # (H, W, 3) uint8
        self._last_frame_rgb = frame_rgb
        frame_tensor = torch.from_numpy(frame_rgb).unsqueeze(0)  # (1, H, W, 3)
        processed = self._pipeline.process(frame_tensor)  # (1, C, H, W)
        with torch.no_grad():
            vis_features = self._encoder.extract(processed.to(self._device))  # (1, D)
        return torch.cat([vis_features.squeeze(0), q_meas], dim=0).unsqueeze(0)

    @property
    def last_frame_rgb(self) -> Optional[np.ndarray]:
        return self._last_frame_rgb


class AsyncVisionJointObsBuilder:
    """Non-blocking variant of :class:`VisionJointObsBuilder`.

    A daemon thread continuously captures camera frames and runs the vision
    encoder; :meth:`build` returns immediately using the most recently cached
    features concatenated with the current joint reading.

    This decouples the slow camera + encoder pipeline from the control loop.
    Vision features may be up to one encoder cycle old, but the control loop
    is no longer blocked on image processing.
    """

    def __init__(self, inner: VisionJointObsBuilder) -> None:
        self._inner = inner
        self._features: Optional[torch.Tensor] = None  # (D,)
        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="obs_async"
        )

    def start(self) -> None:
        """Start background capture. Blocks until the first features are ready."""
        self._thread.start()
        self._ready.wait()

    def _loop(self) -> None:
        inner = self._inner
        while not self._stop_event.is_set():
            frame_rgb = inner._camera.get_frame_rgb()
            frame_tensor = torch.from_numpy(frame_rgb).unsqueeze(0)
            processed = inner._pipeline.process(frame_tensor)
            with torch.no_grad():
                features = inner._encoder.extract(processed.to(inner._device)).squeeze(
                    0
                )  # (D,)
            with self._lock:
                self._features = features
                self._last_frame = frame_rgb
                inner._last_frame_rgb = frame_rgb
            self._ready.set()

    def build(self, q_meas: torch.Tensor) -> torch.Tensor:
        """Return cached obs — no camera I/O, no encoding."""
        with self._lock:
            features = self._features
        if features is None:
            raise RuntimeError("AsyncVisionJointObsBuilder: no features cached yet.")
        return torch.cat([features, q_meas], dim=0).unsqueeze(0)

    @property
    def last_frame_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._last_frame

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)


# ── Inference loop ────────────────────────────────────────────────────────────


# A Policy is any callable matching so101_real.policies.Policy.
PolicyFn = Callable[[torch.Tensor], torch.Tensor]


class InferenceLoop:
    """Policy-agnostic real-robot inference loop.

    Parameters
    ----------
    robot:
        Connected :class:`So101Robot` (caller manages lifecycle).
    robot_config:
        Loaded :class:`RobotConfig` — used for reset pose + ``max_delta_rad``.
    ctrl_config:
        Loaded :class:`ControllerConfig`.
    policy:
        Any callable returning a normalized action in ``[-1, 1]`` of shape
        ``(1, n_act)``.  See :mod:`so101_real.policies`.
    obs_builder:
        Constructs the per-tick observation passed to ``policy``.
    joint_lower_rad / joint_upper_rad:
        Canonical-radian bounds used for the action → joint-target mapping.  Must
        have length ``n_joints``.
    recorder:
        Optional :class:`EpisodeRecorder`.  Only useful when ``obs_builder``
        produces camera frames (i.e. :class:`VisionJointObsBuilder`).
    overlay:
        Optional overlay renderer.  Same caveat as ``recorder``.
    ros_publisher:
        Optional ROS2 publisher for the digital twin.
    dry_run:
        If True, skip actuation (``send_joints`` is a no-op).
    """

    def __init__(
        self,
        robot: So101Robot,
        robot_config: RobotConfig,
        ctrl_config: ControllerConfig,
        policy: PolicyFn,
        obs_builder: ObsBuilder,
        joint_lower_rad: list[float],
        joint_upper_rad: list[float],
        recorder: Optional[EpisodeRecorder] = None,
        overlay=None,
        ros_publisher: Optional[RosPublisher] = None,
        dry_run: bool = False,
    ) -> None:
        if len(joint_lower_rad) != len(joint_upper_rad):
            raise ValueError(
                "joint_lower_rad and joint_upper_rad length mismatch: "
                f"{len(joint_lower_rad)} vs {len(joint_upper_rad)}."
            )

        self._robot = robot
        self._robot_config = robot_config
        self._ctrl = ctrl_config
        self._policy = policy
        # Wrap vision obs builders in an async cache so the slow camera +
        # encoder pipeline runs in a background thread and does not block the
        # control loop.  start() blocks until the first features are ready so
        # the loop never begins with a missing observation.
        self._async_obs_builder: Optional[AsyncVisionJointObsBuilder] = None
        if isinstance(obs_builder, VisionJointObsBuilder):
            async_builder = AsyncVisionJointObsBuilder(obs_builder)
            async_builder.start()
            self._obs_builder = async_builder
            self._async_obs_builder = async_builder
        else:
            self._obs_builder = obs_builder
        self._recorder = recorder
        self._overlay = overlay
        self._ros_publisher = ros_publisher
        self._dry_run = dry_run

        device = torch.device(ctrl_config.device)
        self._device = device

        self._safety = SafetyLayer(
            joint_lower_rad=joint_lower_rad,
            joint_upper_rad=joint_upper_rad,
            max_delta_rad=robot_config.max_delta_rad,
        )
        # JointUnitConverter owns the normalized ↔ canonical-radian mapping.
        # Joint names are not needed here (vectors only); use placeholders.
        self._units = JointUnitConverter(
            joint_names=[f"j{i}" for i in range(len(joint_lower_rad))],
            lower_rad=joint_lower_rad,
            upper_rad=joint_upper_rad,
        )
        self._n_joints = len(joint_lower_rad)
        self._ema_target: Optional[torch.Tensor] = None

    def run(
        self,
        episodes: int,
        seed: Optional[int] = None,
        max_steps_per_episode: Optional[int] = None,
    ) -> None:
        """Run the inference loop for a fixed number of episodes.

        Parameters
        ----------
        episodes:
            Number of episodes to execute.
        seed:
            Optional RNG seed for reproducibility.
        max_steps_per_episode:
            If set, terminate each episode after this many ticks (in addition
            to the usual Ctrl-C interrupt).  Useful for the static-policy
            "hold for N seconds" use case.
        """
        if seed is not None:
            torch.manual_seed(seed)

        control_hz = self._ctrl.control_hz
        tick_period = 1.0 / control_hz

        print(
            f"[InferenceLoop] Starting {episodes} episode(s) at {control_hz:.1f} Hz "
            f"({'DRY RUN' if self._dry_run else 'LIVE'})"
        )

        for ep_idx in range(episodes):
            print(f"[InferenceLoop] Episode {ep_idx + 1}/{episodes}")
            self._reset_to_start_pose(control_hz)
            self._ema_target = None  # Reset EMA at episode start
            self._run_episode(ep_idx, tick_period, max_steps_per_episode)
            if self._recorder is not None:
                self._recorder.end_episode()

        print("[InferenceLoop] All episodes complete.")

    def reset_to_start_pose(self) -> None:
        """Move the arm to the configured reset pose (see ``robot.yaml::reset_pose``).

        A no-op when ``reset_pose`` is absent, disabled, or in dry-run mode.
        """
        self._reset_to_start_pose(self._ctrl.control_hz)

    def _reset_to_start_pose(self, control_hz: float) -> None:
        """Move the arm to the configured start pose at the beginning of an episode.

        Interpolates linearly from the current joint positions to the target
        over ``reset_pose.duration_s`` seconds, sending one command per control
        tick.  Skipped if ``reset_pose`` is absent or ``enabled: false`` in
        robot.yaml.  Also skipped in dry-run mode.
        """
        rp: Optional[ResetPoseCfg] = self._robot_config.reset_pose
        if rp is None or not rp.enabled or self._dry_run:
            return

        n_joints = len(rp.joints_rad)
        target = torch.tensor(rp.joints_rad, dtype=torch.float32)
        n_steps = max(1, round(rp.duration_s * control_hz))
        tick_period = 1.0 / control_hz

        q_start = self._robot.read_joints().cpu().float()
        if q_start.shape[0] != n_joints:
            raise ValueError(
                f"reset_pose.joints_rad has {n_joints} entries but robot has "
                f"{q_start.shape[0]} joints.  Update robot.yaml to match."
            )

        print(
            f"[InferenceLoop] Resetting to start pose over {rp.duration_s:.1f}s "
            f"({n_steps} steps)..."
        )
        for step in range(n_steps):
            t = (step + 1) / n_steps  # linear interpolation parameter in (0, 1]
            q_cmd = q_start + t * (target - q_start)
            self._robot.send_joints(q_cmd)
            time.sleep(tick_period)
        print("[InferenceLoop] Start pose reached.")

    def _run_episode(
        self,
        episode_id: int,
        tick_period: float,
        max_steps: Optional[int],
    ) -> None:
        """Run a single episode until interrupted or ``max_steps`` reached."""
        step = 0
        try:
            while max_steps is None or step < max_steps:
                t_start = time.monotonic()
                self._tick(episode_id)
                step += 1
                elapsed = time.monotonic() - t_start
                remaining = tick_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print(f"\n[InferenceLoop] Episode {episode_id} interrupted.")
            raise

    def _tick(self, episode_id: int) -> None:
        """Execute one control step."""
        # 1. Read joint positions
        q_meas = self._robot.read_joints().to(self._device)  # (n_joints,)

        # 1a. Publish to digital twin (ROS2)
        if self._ros_publisher is not None:
            self._ros_publisher.publish(q_meas)

        # 2. Build observation
        obs = self._obs_builder.build(q_meas)  # (1, obs_dim)

        # 3. Policy forward pass — returns (1, n_act) in [-1, 1]
        with torch.no_grad():
            action = self._policy(obs).squeeze(0)  # (n_act,)

        # 4. Map action ∈ [-1, 1] → canonical-radian joint targets.
        # Policies may emit values slightly outside [-1, 1]; the safety layer
        # below clamps the resulting joint targets so we skip validate_norm.
        q_target = self._units.normalized_to_canonical(action)

        # 5. EMA smoothing
        if self._ema_target is None:
            self._ema_target = q_target
        else:
            alpha = self._ctrl.ema_alpha
            self._ema_target = alpha * q_target + (1.0 - alpha) * self._ema_target
        q_smooth = self._ema_target

        # 6. Safety layer
        q_safe = self._safety.apply(q_smooth, q_meas)

        # 7. Send to robot
        if not self._dry_run:
            self._robot.send_joints(q_safe)

        # 8. Record (requires a camera frame from the obs builder)
        if self._recorder is not None:
            frame_rgb = self._obs_builder.last_frame_rgb
            if frame_rgb is None:
                raise RuntimeError(
                    "EpisodeRecorder is enabled but the active ObsBuilder does "
                    "not produce camera frames.  Pair the recorder with "
                    "VisionJointObsBuilder, or disable recording."
                )
            self._recorder.record_step(
                frame_rgb=frame_rgb,
                joint_positions_rad=q_meas,
                joint_targets_rad=q_safe,
                actions_raw=action,
                episode_id=episode_id,
            )

        # 9. Update overlay (only meaningful when there is a camera frame)
        if self._overlay is not None:
            frame_rgb = self._obs_builder.last_frame_rgb
            if frame_rgb is not None:
                self._overlay.update(frame_rgb, q_meas, q_safe, action)

    def destroy(self) -> None:
        """Release resources (async obs builder and ROS2 publisher, if active)."""
        if self._async_obs_builder is not None:
            self._async_obs_builder.stop()
            self._async_obs_builder = None
        if self._ros_publisher is not None:
            self._ros_publisher.destroy()
            self._ros_publisher = None

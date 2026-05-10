"""controller.py — Real-robot inference loop.

Runs at ``control_hz`` Hz from the bundle manifest.  Each tick:
  1. Grab camera frame
  2. Run image pipeline
  3. Extract vision features
  4. Build obs: [vision_features | joint_positions | zeros for actor_metrics]
  5. Forward pass through policy
  6. Map action → joint targets (same formula as training env)
  7. Apply safety layer
  8. Apply EMA smoothing
  9. Send to robot
  10. Record (optional)
  11. Update overlay (optional)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from .bundle import DeployBundle
from .camera import CameraSource
from .image_pipeline import build_deploy_pipeline
from .policy import load_policy
from .recorder import EpisodeRecorder
from .robot import RobotConfig, So101Robot, ResetPoseCfg
from .ros_publisher import RosPublisher
from .safety import SafetyLayer
from .vision import build_vision_encoder


@dataclass
class ControllerConfig:
    """Explicit inference loop configuration.

    Load from YAML via ``ControllerConfig.load(path)``.
    """

    ema_alpha: float
    """EMA smoothing coefficient for joint targets. 1.0 = no smoothing, 0.0 = freeze."""

    device: str
    """PyTorch device string for policy and vision encoder inference."""

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
        required = {"ema_alpha", "device"}
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
        return cls(
            ema_alpha=ema_alpha,
            device=str(ctrl["device"]),
        )


class InferenceLoop:
    """Real-robot inference loop.

    Parameters
    ----------
    bundle:
        Validated deploy bundle.
    camera:
        Open CameraSource (caller manages lifecycle).
    robot:
        Connected So101Robot (caller manages lifecycle).
    robot_config:
        Loaded RobotConfig.
    ctrl_config:
        Loaded ControllerConfig.
    recorder:
        Optional EpisodeRecorder for logging.
    overlay:
        Optional OverlayRenderer for live visualization.
    ros_publisher:
        Optional RosPublisher for streaming joint states to the digital twin.
    dry_run:
        If True, skip actuation (send_joints is a no-op).
    """

    def __init__(
        self,
        bundle: DeployBundle,
        camera: CameraSource,
        robot: So101Robot,
        robot_config: RobotConfig,
        ctrl_config: ControllerConfig,
        recorder: Optional[EpisodeRecorder] = None,
        overlay=None,
        ros_publisher: Optional[RosPublisher] = None,
        dry_run: bool = False,
    ) -> None:
        self._bundle = bundle
        self._camera = camera
        self._robot = robot
        self._robot_config = robot_config
        self._ctrl = ctrl_config
        self._recorder = recorder
        self._overlay = overlay
        self._dry_run = dry_run
        self._ros_publisher = ros_publisher

        device = torch.device(ctrl_config.device)

        # Build all inference components
        self._pipeline = build_deploy_pipeline(bundle)
        self._encoder = build_vision_encoder(bundle, device=device)
        self._policy = load_policy(bundle, device=device)

        # Safety layer uses bundle's joint limits + robot config's max_delta
        self._safety = SafetyLayer(
            joint_lower_rad=bundle.joint_lower_rad,
            joint_upper_rad=bundle.joint_upper_rad,
            max_delta_rad=robot_config.max_delta_rad,
        )

        self._lower = torch.tensor(
            bundle.joint_lower_rad, dtype=torch.float32, device=device
        )
        self._upper = torch.tensor(
            bundle.joint_upper_rad, dtype=torch.float32, device=device
        )
        self._device = device
        self._ema_target: Optional[torch.Tensor] = None

        # actor_obs_metrics: validated empty at export time; raise clearly if present
        if bundle.actor_obs_metrics:
            raise ValueError(
                f"Deploy bundle contains actor_obs_metrics={bundle.actor_obs_metrics} "
                "which are not supported at deploy time.\n"
                "Re-export the bundle after removing these metrics from the env config, "
                "or add support for them in so101_real/controller.py."
            )

    def run(self, episodes: int, seed: Optional[int] = None) -> None:
        """Run the inference loop for a fixed number of episodes.

        Parameters
        ----------
        episodes:
            Number of episodes to execute.
        seed:
            Optional RNG seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)

        control_hz = self._bundle.control_hz
        tick_period = 1.0 / control_hz

        print(
            f"[InferenceLoop] Starting {episodes} episode(s) at {control_hz:.1f} Hz "
            f"({'DRY RUN' if self._dry_run else 'LIVE'})"
        )

        for ep_idx in range(episodes):
            print(f"[InferenceLoop] Episode {ep_idx + 1}/{episodes}")
            self._reset_to_start_pose(control_hz)
            self._ema_target = None  # Reset EMA at episode start
            self._run_episode(ep_idx, tick_period)
            if self._recorder is not None:
                self._recorder.end_episode()

        print("[InferenceLoop] All episodes complete.")

    def _reset_to_start_pose(self, control_hz: float) -> None:
        """Move the arm to the configured start pose at the beginning of an episode.

        Interpolates linearly from the current joint positions to the target over
        ``reset_pose.duration_s`` seconds, sending one command per control tick.
        Skipped if ``reset_pose`` is absent or ``enabled: false`` in robot.yaml.
        Also skipped in dry-run mode.
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
                f"{q_start.shape[0]} joints.  Update robot.yaml to match the bundle."
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

    def _run_episode(self, episode_id: int, tick_period: float) -> None:
        """Run a single episode until interrupted."""
        try:
            while True:
                t_start = time.monotonic()
                self._tick(episode_id)
                elapsed = time.monotonic() - t_start
                remaining = tick_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print(f"\n[InferenceLoop] Episode {episode_id} interrupted.")
            raise

    def _tick(self, episode_id: int) -> None:
        """Execute one control step."""
        # 1. Grab frame
        frame_rgb = self._camera.get_frame_rgb()  # (H, W, 3) uint8 numpy

        # 2. Run image pipeline  (N, H, W, C) uint8 → (N, C, H, W) float
        frame_tensor = torch.from_numpy(frame_rgb).unsqueeze(0)  # (1, H, W, 3)
        processed = self._pipeline.process(frame_tensor)  # (1, C, H, W)

        # 3. Extract vision features
        with torch.no_grad():
            vis_features = self._encoder.extract(processed.to(self._device))  # (1, D)

        # 4. Read joint positions
        q_meas = self._robot.read_joints().to(self._device)  # (n_joints,)

        # 4a. Publish to digital twin (ROS2) — no-op if ros_publisher is None
        if self._ros_publisher is not None:
            self._ros_publisher.publish(q_meas)

        # 5. Build obs: [vision_features | q]
        # actor_obs_metrics are validated empty at InferenceLoop.__init__
        obs = torch.cat([vis_features.squeeze(0), q_meas], dim=0).unsqueeze(
            0
        )  # (1, obs_dim)

        # 6. Policy forward pass
        with torch.no_grad():
            action = self._policy(obs).squeeze(0)  # (act_dim,)

        # 7. Map action ∈ [-1, 1] → joint targets (same as training env)
        t = 0.5 * (action + 1.0)
        q_target = self._lower + t * (self._upper - self._lower)

        # 8. EMA smoothing
        if self._ema_target is None:
            self._ema_target = q_target
        else:
            alpha = self._ctrl.ema_alpha
            self._ema_target = alpha * q_target + (1.0 - alpha) * self._ema_target
        q_smooth = self._ema_target

        # 9. Safety layer
        q_safe = self._safety.apply(q_smooth, q_meas)

        # 10. Send to robot
        if not self._dry_run:
            self._robot.send_joints(q_safe)

        # 11. Record
        if self._recorder is not None:
            self._recorder.record_step(
                frame_rgb=frame_rgb,
                joint_positions_rad=q_meas,
                joint_targets_rad=q_safe,
                actions_raw=action,
                episode_id=episode_id,
            )

        # 12. Update overlay
        if self._overlay is not None:
            self._overlay.update(frame_rgb, q_meas, q_safe, action)

    def destroy(self) -> None:
        """Release resources (ROS2 publisher, if active)."""
        if self._ros_publisher is not None:
            self._ros_publisher.destroy()
            self._ros_publisher = None

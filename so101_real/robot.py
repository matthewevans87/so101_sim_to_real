"""robot.py — SO-101 robot interface for real-robot inference.

Wraps the LeRobot SO101Follower API.  External communication uses degrees
(LeRobot convention); internally this codebase works in **canonical radians**
(zero-at-home, matching the URDF the policy was trained against).  There is
only one internal frame; no simulator is involved at runtime.

Joint calibration
-----------------
LeRobot's per-motor calibration uses its own zero-points and scales that do
not match the canonical (URDF) frame.  The ``joint_calibration`` section of
``robot.yaml`` corrects this via a per-joint linear transform applied after
reading and before writing joint positions::

    q_rad     = scale * q_lerobot_rad + offset_rad          # read
    q_lerobot = (q_rad - offset_rad) / scale                 # send

Only joints listed in ``joint_calibration`` are corrected; unlisted joints
use the identity transform (scale=1.0, offset=0.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from .units import JointUnitConverter, from_robot_config


@dataclass
class JointLimitEntry:
    """Physical range of a single joint in canonical radians."""

    lower_rad: float
    """Lower joint limit in canonical radians."""

    upper_rad: float
    """Upper joint limit in canonical radians."""


@dataclass
class JointCalibrationEntry:
    """Per-joint linear map from LeRobot's native frame to canonical radians.

    Applied on read as ``q_rad = scale * q_lerobot_rad + offset_rad`` and
    inverted on send.  A negative ``scale`` indicates the LeRobot motor
    rotates in the opposite direction to the canonical (URDF) convention.
    """

    scale: float
    """Multiplier applied to the LeRobot radian value when reading."""

    offset_rad: float
    """Additive offset (radians) applied after scaling when reading."""


@dataclass
class ResetPoseCfg:
    """Configuration for moving the arm to a fixed starting pose at episode start."""

    enabled: bool
    """When False the reset step is skipped entirely."""

    joints_rad: list
    """Target joint angles in radians, aligned with the bundle joint order."""

    duration_s: float
    """Seconds over which to linearly interpolate from current to target pose."""


@dataclass
class RobotConfig:
    """Explicit robot configuration — all fields required, no silent defaults.

    Load from YAML via ``RobotConfig.load(path)``.
    """

    port: str
    """Serial port of the SO-101 follower arm (e.g. /dev/ttyUSB0)."""

    calibration_file: str
    """Path to the LeRobot calibration JSON file."""

    max_delta_rad: float
    """Maximum allowed joint displacement per control step (radians).

    Clips the difference between commanded and current joint positions.
    Prevents large sudden movements when the policy outputs an extreme action.
    """

    reset_pose: Optional[ResetPoseCfg]
    """Optional reset pose applied at the start of every episode.  ``None`` if the
    ``reset_pose`` section is absent from the config file."""

    joint_calibration: dict[str, JointCalibrationEntry]
    """Per-joint linear transforms correcting LeRobot's native frame into
    canonical radians.  Keyed by joint name; missing joints use identity."""

    joint_limits: dict[str, JointLimitEntry]
    """Physical range of each joint in canonical radians.  Required in
    robot.yaml; used for normalised-action mapping and safety checks."""

    def joint_bounds(self) -> tuple[list[str], list[float], list[float]]:
        """Return ``(joint_names, lower_rad, upper_rad)`` in YAML iteration order."""
        names: list[str] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for name, entry in self.joint_limits.items():
            names.append(name)
            lowers.append(entry.lower_rad)
            uppers.append(entry.upper_rad)
        return names, lowers, uppers

    @classmethod
    def load(cls, path: str | Path) -> "RobotConfig":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Robot config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Reject legacy YAML keys explicitly so stale configs fail loudly.
        if "sim_joint_limits" in data:
            raise ValueError(
                f"Robot config uses the legacy key 'sim_joint_limits'.  Rename it"
                f" to 'joint_limits' (same schema). Config path: {path}"
            )

        robot = data.get("robot")
        if robot is None:
            raise ValueError(
                f"Robot config YAML must contain a top-level 'robot' key: {path}"
            )
        required = {"port", "calibration_file", "max_delta_rad"}
        missing = required - set(robot)
        if missing:
            raise ValueError(
                f"Robot config is missing required keys: {sorted(missing)}\n"
                f"Config path: {path}"
            )

        reset_pose: Optional[ResetPoseCfg] = None
        rp_data = data.get("reset_pose")
        if rp_data is not None:
            required_rp = {"enabled", "joints_rad", "duration_s"}
            missing_rp = required_rp - set(rp_data)
            if missing_rp:
                raise ValueError(
                    f"reset_pose config is missing required keys: {sorted(missing_rp)}\n"
                    f"Config path: {path}"
                )
            reset_pose = ResetPoseCfg(
                enabled=bool(rp_data["enabled"]),
                joints_rad=[float(v) for v in rp_data["joints_rad"]],
                duration_s=float(rp_data["duration_s"]),
            )

        # Parse joint_calibration (optional; defaults to identity for all joints)
        joint_calibration: dict[str, JointCalibrationEntry] = {}
        cal_data = data.get("joint_calibration") or {}
        for joint_name, entry in cal_data.items():
            # Reject legacy per-entry keys.
            legacy = {"lero_to_sim_scale", "lero_to_sim_offset_rad"} & set(entry)
            if legacy:
                raise ValueError(
                    f"joint_calibration[{joint_name!r}] uses legacy key(s) "
                    f"{sorted(legacy)}. Rename 'lero_to_sim_scale' -> 'scale' and "
                    f"'lero_to_sim_offset_rad' -> 'offset_rad'. Config path: {path}"
                )
            required_cal = {"scale", "offset_rad"}
            missing_cal = required_cal - set(entry)
            if missing_cal:
                raise ValueError(
                    f"joint_calibration[{joint_name!r}] is missing required keys: "
                    f"{sorted(missing_cal)}\nConfig path: {path}"
                )
            joint_calibration[str(joint_name)] = JointCalibrationEntry(
                scale=float(entry["scale"]),
                offset_rad=float(entry["offset_rad"]),
            )

        # Parse joint_limits (required)
        jl_raw = data.get("joint_limits")
        if not jl_raw:
            raise ValueError(
                f"robot.yaml is missing a required 'joint_limits' section.\n"
                f"Config path: {path}"
            )
        joint_limits: dict[str, JointLimitEntry] = {}
        for jname, jentry in jl_raw.items():
            missing_jl = {"lower_rad", "upper_rad"} - set(jentry)
            if missing_jl:
                raise ValueError(
                    f"joint_limits[{jname!r}] is missing required keys: "
                    f"{sorted(missing_jl)}\nConfig path: {path}"
                )
            joint_limits[str(jname)] = JointLimitEntry(
                lower_rad=float(jentry["lower_rad"]),
                upper_rad=float(jentry["upper_rad"]),
            )

        return cls(
            port=str(robot["port"]),
            calibration_file=str(robot["calibration_file"]),
            max_delta_rad=float(robot["max_delta_rad"]),
            reset_pose=reset_pose,
            joint_calibration=joint_calibration,
            joint_limits=joint_limits,
        )


class So101Robot:
    """LeRobot SO101Follower wrapper.

    Converts between radians (internal, training convention) and degrees
    (LeRobot API convention).

    Usage::

        robot = So101Robot(config, joint_names)
        robot.connect()
        q = robot.read_joints()          # tensor of shape (n_joints,) in radians
        robot.send_joints(q_target_rad)  # send joint targets in radians
        robot.disconnect()
    """

    def __init__(self, config: RobotConfig, joint_names: list[str]) -> None:
        self._cfg = config
        self._joint_names = joint_names
        self._follower: Optional[object] = None
        # Single source of truth for canonical ↔ LeRobot conversions.
        # All rad/deg/lrad/ldeg arithmetic lives in JointUnitConverter; this
        # class only knows about reading degrees off the bus and writing
        # degrees back to it.
        self._units = from_robot_config(
            joint_names=joint_names,
            joint_calibration=config.joint_calibration,
        )

        if config.joint_calibration:
            print(
                f"[So101Robot] Joint calibrations active: {list(config.joint_calibration)}"
            )

    @property
    def units(self) -> "JointUnitConverter":
        """Per-joint unit converter (canonical ↔ LeRobot)."""
        return self._units

    @property
    def _cal_scale(self) -> torch.Tensor:
        """Backwards-compatible accessor: ``q_rad = scale * q_lrad + offset``."""
        return self._units.lero_scale

    @property
    def _cal_offset(self) -> torch.Tensor:
        """Backwards-compatible accessor: ``q_rad = scale * q_lrad + offset``."""
        return self._units.lero_offset_rad

    def connect(self, dry_run: bool = False) -> None:
        """Open the serial connection to the robot.

        Parameters
        ----------
        dry_run:
            If True, skip actual hardware connection (for testing pipelines
            without physical hardware).
        """
        if dry_run:
            print("[So101Robot] dry_run=True — skipping hardware connection.")
            self._follower = None
            return

        try:
            # lerobot >= 0.5: SO100/SO101 are unified under `so_follower`.
            # `SO101Follower` / `SO101FollowerConfig` are kept as aliases of
            # `SOFollower` / `SOFollowerRobotConfig` for backwards compatibility.
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise ImportError(
                "lerobot >= 0.5 is required for real-robot deployment.\n"
                "Install with: pip install 'lerobot[feetech]'\n"
                f"Original error: {exc}"
            ) from exc

        calibration_path = Path(self._cfg.calibration_file).expanduser().resolve()
        # NOTE: We intentionally pass max_relative_target=None to LeRobot.
        # LeRobot's per-tick clamp uses (goal - present) and, when the joint
        # is far from goal (e.g. shoulder_lift sagged past its lower limit
        # under gravity), forces the Feetech PID into a low-error/low-torque
        # regime that cannot overcome the gravity load -- the arm gets stuck.
        # Per-tick motion limiting is handled upstream by our SafetyLayer
        # (max_delta_rad) for closed-loop control, and by reset_pose's linear
        # interpolation for startup recovery.
        robot_cfg = SO101FollowerConfig(
            port=self._cfg.port,
            id=calibration_path.stem,
            calibration_dir=calibration_path.parent,
            max_relative_target=None,
            use_degrees=True,
        )
        follower = SO101Follower(robot_cfg)
        follower.connect(calibrate=False)
        self._follower = follower
        print(f"[So101Robot] Connected on {self._cfg.port}")

    def read_joints(self) -> torch.Tensor:
        """Read current joint positions.

        Returns
        -------
        torch.Tensor
            Shape ``(n_joints,)`` float32, values in **canonical radians**.
        """
        if self._follower is None:
            # dry_run mode — return zeros
            return torch.zeros(len(self._joint_names))

        obs = self._follower.get_observation()
        positions_deg = [obs[f"{name}.pos"] for name in self._joint_names]
        q_lrad = torch.tensor(
            [math.radians(d) for d in positions_deg], dtype=torch.float32
        )
        return self._units.lero_rad_to_canonical(q_lrad)

    def send_joints(self, q_target_rad: torch.Tensor) -> None:
        """Send joint position targets.

        Applies the inverse calibration ``q_lerobot = (q_rad - offset) / scale``
        and forwards the result to LeRobot's ``send_action``.  Per-tick safety
        clamping is **not** applied here: it is the caller's responsibility
        (``SafetyLayer`` in the closed-loop controller, linear interpolation
        in ``reset_pose``).

        Parameters
        ----------
        q_target_rad:
            Shape ``(n_joints,)`` float32, values in **canonical radians**.
        """
        if self._follower is None:
            return  # dry_run mode

        q_np = q_target_rad.detach().cpu().float()
        q_lerobot = self._units.canonical_to_lero_rad(q_np)
        action = {
            f"{name}.pos": math.degrees(float(q_lerobot[i]))
            for i, name in enumerate(self._joint_names)
        }
        self._follower.send_action(action)

    def disconnect(self) -> None:
        if self._follower is not None:
            try:
                self._follower.disconnect()
            except Exception as exc:
                print(
                    f"[So101Robot] Warning: error during disconnect (torque may still be enabled): {exc}"
                )
            finally:
                self._follower = None

    def __enter__(self) -> "So101Robot":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

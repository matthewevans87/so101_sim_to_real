"""robot.py — SO-101 robot interface for real-robot inference.

Wraps the LeRobot SO101Follower API.  All external communication uses
degrees (LeRobot convention).  Internally we work in radians to match
the training obs/action contract.

Joint calibration
-----------------
LeRobot and the Isaac Sim USD may use different zero-points or scales for
some joints.  The ``joint_calibration`` section of ``robot.yaml`` corrects
for these differences via a per-joint linear transform applied after reading
and before writing joint positions::

    q_sim  = lero_to_sim_scale * q_lerobot + lero_to_sim_offset_rad   # read
    q_lerobot = (q_sim - lero_to_sim_offset_rad) / lero_to_sim_scale  # send

Only joints listed in ``joint_calibration`` are corrected; unlisted joints
use the identity transform (scale=1.0, offset=0.0).

Wrap handling (multi-turn joints, e.g. wrist_roll)
--------------------------------------------------
A joint may additionally specify ``wrap_period_rad`` (and optionally
``lero_branch_center_rad``) to declare that its raw LeRobot reading lives
on a circle of period :math:`P` (typically :math:`2\\pi` for one full encoder
turn).  When set:

* ``read_joints()`` shifts the raw reading by an integer multiple of :math:`P`
  to land closest to ``lero_branch_center_rad`` (or, if no prior reading,
  the configured center).  This produces a continuous sim-space coordinate
  even as the encoder crosses its wrap boundary.
* ``send_joints()`` shifts the commanded LeRobot value by an integer multiple
  of :math:`P` to land closest to the **most recent raw reading**, so the
  servo never takes the long way around.  ``send_joints()`` therefore
  requires that ``read_joints()`` was called at least once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml


@dataclass
class JointCalibrationEntry:
    """Per-joint linear calibration between LeRobot and sim (PhysX) conventions."""

    lero_to_sim_scale: float
    """Multiplier applied to the LeRobot radian value when reading."""

    lero_to_sim_offset_rad: float
    """Additive offset (radians) applied after scaling when reading."""

    wrap_period_rad: Optional[float] = None
    """If not None, treat the raw LeRobot reading as a value on a circle of this
    period (radians).  Typically ``2*pi`` for a single-turn encoder.  ``None``
    disables wrap handling for this joint."""

    lero_branch_center_rad: Optional[float] = None
    """Anchor for the read-side unwrap when no prior reading is available.
    Required when ``wrap_period_rad`` is set; ignored otherwise."""

    @property
    def sim_to_lero_scale(self) -> float:
        return 1.0 / self.lero_to_sim_scale

    @property
    def sim_to_lero_offset_rad(self) -> float:
        return -self.lero_to_sim_offset_rad / self.lero_to_sim_scale


def unwrap_to_branch(q: float, period: float, center: float) -> float:
    """Shift ``q`` by the integer multiple of ``period`` that minimises
    ``|q + k*period - center|``.

    Vectorised callers should use the inline expression directly; this helper
    is provided for clarity and for use in calibration scripts.
    """
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    k = round((center - q) / period)
    return q + k * period


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
    """Per-joint linear transforms correcting for convention differences between
    LeRobot and the sim.  Keyed by joint name; missing joints use identity."""

    @classmethod
    def load(cls, path: str | Path) -> "RobotConfig":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Robot config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
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
            required_cal = {"lero_to_sim_scale", "lero_to_sim_offset_rad"}
            missing_cal = required_cal - set(entry)
            if missing_cal:
                raise ValueError(
                    f"joint_calibration[{joint_name!r}] is missing required keys: "
                    f"{sorted(missing_cal)}\nConfig path: {path}"
                )
            wrap_period = entry.get("wrap_period_rad", None)
            branch_center = entry.get("lero_branch_center_rad", None)
            if wrap_period is not None:
                wrap_period = float(wrap_period)
                if wrap_period <= 0.0:
                    raise ValueError(
                        f"joint_calibration[{joint_name!r}].wrap_period_rad must be positive, "
                        f"got {wrap_period}.\nConfig path: {path}"
                    )
                if branch_center is None:
                    raise ValueError(
                        f"joint_calibration[{joint_name!r}] sets wrap_period_rad but is "
                        f"missing lero_branch_center_rad (required when wrap is enabled).\n"
                        f"Config path: {path}"
                    )
                branch_center = float(branch_center)
            elif branch_center is not None:
                raise ValueError(
                    f"joint_calibration[{joint_name!r}] sets lero_branch_center_rad but "
                    f"wrap_period_rad is not set; both must be provided together or omitted.\n"
                    f"Config path: {path}"
                )
            joint_calibration[str(joint_name)] = JointCalibrationEntry(
                lero_to_sim_scale=float(entry["lero_to_sim_scale"]),
                lero_to_sim_offset_rad=float(entry["lero_to_sim_offset_rad"]),
                wrap_period_rad=wrap_period,
                lero_branch_center_rad=branch_center,
            )

        return cls(
            port=str(robot["port"]),
            calibration_file=str(robot["calibration_file"]),
            max_delta_rad=float(robot["max_delta_rad"]),
            reset_pose=reset_pose,
            joint_calibration=joint_calibration,
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
        # Pre-build per-joint correction tensors (identity where unconfigured)
        n = len(joint_names)
        scales = [
            config.joint_calibration.get(
                name, JointCalibrationEntry(1.0, 0.0)
            ).lero_to_sim_scale
            for name in joint_names
        ]
        offsets = [
            config.joint_calibration.get(
                name, JointCalibrationEntry(1.0, 0.0)
            ).lero_to_sim_offset_rad
            for name in joint_names
        ]
        self._cal_scale = torch.tensor(scales, dtype=torch.float32)  # lero → sim
        self._cal_offset = torch.tensor(offsets, dtype=torch.float32)  # lero → sim

        # Per-joint wrap configuration.  NaN sentinel = wrap disabled for that joint.
        # Keeping NaN avoids a parallel mask tensor while letting torch.isfinite() check it.
        wrap_periods = [
            (
                config.joint_calibration[name].wrap_period_rad
                if name in config.joint_calibration
                and config.joint_calibration[name].wrap_period_rad is not None
                else float("nan")
            )
            for name in joint_names
        ]
        wrap_centers = [
            (
                config.joint_calibration[name].lero_branch_center_rad
                if name in config.joint_calibration
                and config.joint_calibration[name].lero_branch_center_rad is not None
                else float("nan")
            )
            for name in joint_names
        ]
        self._wrap_period = torch.tensor(wrap_periods, dtype=torch.float32)
        self._wrap_center = torch.tensor(wrap_centers, dtype=torch.float32)
        self._has_wrap = bool(torch.isfinite(self._wrap_period).any().item())

        # Cache of the most recent **unwrapped raw LeRobot** reading (radians).
        # Populated by read_joints(); required by send_joints() for any wrapped joint.
        self._last_raw_lero: Optional[torch.Tensor] = None

        if config.joint_calibration:
            wrapped = [
                name
                for name in joint_names
                if name in config.joint_calibration
                and config.joint_calibration[name].wrap_period_rad is not None
            ]
            print(
                f"[So101Robot] Joint calibrations active: {list(config.joint_calibration)}"
            )
            if wrapped:
                print(f"[So101Robot] Wrap-aware joints: {wrapped}")

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
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise ImportError(
                "lerobot is required for real-robot deployment.\n"
                "Install with: pip install lerobot\n"
                f"Original error: {exc}"
            ) from exc

        calibration_path = Path(self._cfg.calibration_file).expanduser().resolve()
        robot_cfg = SO101FollowerConfig(
            port=self._cfg.port,
            id=calibration_path.stem,
            calibration_dir=calibration_path.parent,
            max_relative_target=math.degrees(self._cfg.max_delta_rad),
            use_degrees=True,
        )
        follower = SO101Follower(robot_cfg)
        follower.connect(calibrate=False)
        self._follower = follower
        print(f"[So101Robot] Connected on {self._cfg.port}")

    def read_joints(self) -> torch.Tensor:
        """Read current joint positions.

        For joints with ``wrap_period_rad`` set, the raw LeRobot reading is
        first unwrapped onto the nearest branch of either (a) the previously
        cached raw reading or (b) the configured ``lero_branch_center_rad``
        on the very first call.  The unwrapped raw value is cached for use
        by the next ``send_joints()`` call.

        Returns
        -------
        torch.Tensor
            Shape ``(n_joints,)`` float32, values in **radians** (sim convention).
        """
        if self._follower is None:
            # dry_run mode — return zeros
            return torch.zeros(len(self._joint_names))

        obs = self._follower.get_observation()
        positions_deg = [obs[f"{name}.pos"] for name in self._joint_names]
        positions_rad = torch.tensor(
            [math.radians(d) for d in positions_deg], dtype=torch.float32
        )

        # Apply wrap unwrap to raw LeRobot values for joints that need it.
        if self._has_wrap:
            anchor = (
                self._last_raw_lero
                if self._last_raw_lero is not None
                else self._wrap_center
            )
            # k = round((anchor - q) / period); q' = q + k*period.
            # NaN periods produce NaN k → mask back to original value.
            period = self._wrap_period
            k = torch.round((anchor - positions_rad) / period)
            unwrapped = positions_rad + k * period
            mask = torch.isfinite(period)
            positions_rad = torch.where(mask, unwrapped, positions_rad)

        # Cache the (possibly unwrapped) raw value for the next send_joints() call.
        self._last_raw_lero = positions_rad.clone()

        # Apply lero → sim calibration: q_sim = scale * q_lerobot + offset
        return self._cal_scale * positions_rad + self._cal_offset

    def send_joints(self, q_target_rad: torch.Tensor) -> None:
        """Send joint position targets.

        For joints with ``wrap_period_rad`` set, the commanded LeRobot value
        is shifted by an integer multiple of the period to land closest to
        the most recent raw reading, ensuring the servo never takes the long
        way around the wrap boundary.  This requires that ``read_joints()``
        was called at least once before the first ``send_joints()``.

        Parameters
        ----------
        q_target_rad:
            Shape ``(n_joints,)`` float32, values in **radians** (sim convention).
        """
        if self._follower is None:
            return  # dry_run mode

        q_np = q_target_rad.detach().cpu().float()
        # Apply inverse sim → lero calibration: q_lerobot = (q_sim - offset) / scale
        q_lerobot = (q_np - self._cal_offset) / self._cal_scale

        if self._has_wrap:
            if self._last_raw_lero is None:
                raise RuntimeError(
                    "send_joints() called before read_joints() while wrap-aware joints "
                    "are configured. The send-side branch projection requires a recent "
                    "raw LeRobot reading. Call read_joints() at least once first."
                )
            period = self._wrap_period
            k = torch.round((self._last_raw_lero - q_lerobot) / period)
            projected = q_lerobot + k * period
            mask = torch.isfinite(period)
            q_lerobot = torch.where(mask, projected, q_lerobot)

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

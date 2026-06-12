"""robot.py — SO-101 robot interface for real-robot inference.

Wraps the LeRobot SO101Follower API.  External communication uses degrees
(LeRobot convention); internally this codebase works in **canonical radians**
(zero-at-home, matching the URDF the policy was trained against).  There is
only one internal frame; no simulator is involved at runtime.

Joint limits schema
-------------------
Each joint in ``robot.yaml::joint_limits`` carries two sub-fields:

``sim: [lower, upper]``
    Canonical URDF radians — the normalization range the policy was trained
    against.  Required for ``unit="norm"`` conversion.  When running
    ``python -m so101_real run``, these values are checked against the
    bundle's embedded limits; pass ``--use-bundle-joint-limits`` to fill them
    automatically from the bundle rather than from ``robot.yaml``.

``real: [lower, upper]``
    LeRobot motor radians at the physical hard stops, measured by
    ``joint_calibrate.py --mode sweep``.  From these two stops the
    calibration transform is derived at parse time::

        scale      = (sim_hi - sim_lo) / (real_hi - real_lo)
        offset_rad = sim_lo - scale * real_lo
        q_rad      = scale * q_lerobot_rad + offset_rad   (read)
        q_lerobot  = (q_rad - offset_rad) / scale          (send)

Both sub-fields are required for hardware-facing commands.  ``sim`` alone
(without ``real``) is accepted when the config is used for sim-only tooling
(``align_camera.py``, ``calibrate_sim_joints.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from so101.utils.units import JointUnitConverter, from_robot_config


@dataclass
class JointLimitEntry:
    """Per-joint limits unifying sim normalization bounds and real-robot calibration.

    The ``sim`` field provides the canonical URDF radian bounds used for
    ``norm`` conversion.  The ``real`` field records the LeRobot motor
    positions (in radians) at the physical hard stops; from these the
    ``lero_scale`` / ``lero_offset_rad`` calibration transform is derived.

    Both fields are optional at the dataclass level so that sim-only tooling
    can load configs without ``real`` data; the ``So101Robot`` constructor
    validates that ``has_real`` is True before connecting to hardware.
    """

    sim_lower_rad: Optional[float]
    """Lower bound in canonical (URDF) radians.  Required for ``unit='norm'``."""

    sim_upper_rad: Optional[float]
    """Upper bound in canonical (URDF) radians.  Required for ``unit='norm'``."""

    real_lower_rad: Optional[float]
    """LeRobot motor position (rad) at the lower physical stop (from sweep)."""

    real_upper_rad: Optional[float]
    """LeRobot motor position (rad) at the upper physical stop (from sweep)."""

    @property
    def has_sim(self) -> bool:
        return self.sim_lower_rad is not None and self.sim_upper_rad is not None

    @property
    def has_real(self) -> bool:
        return self.real_lower_rad is not None and self.real_upper_rad is not None

    @property
    def lower_rad(self) -> float:
        """Canonical lower bound (sim frame).  Raises if ``sim`` is absent."""
        if self.sim_lower_rad is None:
            raise ValueError(
                "joint_limits entry is missing 'sim' bounds. "
                "Add sim: [lower, upper] or pass --use-bundle-joint-limits."
            )
        return self.sim_lower_rad

    @property
    def upper_rad(self) -> float:
        """Canonical upper bound (sim frame).  Raises if ``sim`` is absent."""
        if self.sim_upper_rad is None:
            raise ValueError(
                "joint_limits entry is missing 'sim' bounds. "
                "Add sim: [lower, upper] or pass --use-bundle-joint-limits."
            )
        return self.sim_upper_rad

    @property
    def lero_scale(self) -> float:
        """``scale`` in ``q_rad = scale * q_lrad + offset``.

        Derived from sim and real bounds:
        ``scale = (sim_hi - sim_lo) / (real_hi - real_lo)``.
        Raises if either ``sim`` or ``real`` is absent.
        """
        if not self.has_sim:
            raise ValueError("lero_scale requires 'sim' bounds in joint_limits.")
        if not self.has_real:
            raise ValueError("lero_scale requires 'real' bounds in joint_limits.")
        span_real = self.real_upper_rad - self.real_lower_rad
        if span_real == 0.0:
            raise ValueError(
                "real[upper] == real[lower] — cannot derive scale (zero span)."
            )
        return (self.sim_upper_rad - self.sim_lower_rad) / span_real

    @property
    def lero_offset_rad(self) -> float:
        """``offset_rad`` in ``q_rad = scale * q_lrad + offset``.

        Derived as ``sim_lo - scale * real_lo``.
        """
        return self.sim_lower_rad - self.lero_scale * self.real_lower_rad


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

    max_delta_rad: Optional[float]
    """Maximum allowed joint displacement per control step (radians).

    Clips the difference between commanded and current joint positions.
    Prevents large sudden movements when the policy outputs an extreme action.

    ``None`` means "use the value from the deploy bundle".  Set explicitly in
    robot.yaml to override the bundle value (or to enforce a hardware ceiling).
    """

    reset_pose: Optional[ResetPoseCfg]
    """Optional reset pose applied at the start of every episode.  ``None`` if the
    ``reset_pose`` section is absent from the config file."""

    joint_limits: dict[str, JointLimitEntry]
    """Per-joint limits in the unified ``sim``/``real`` schema.  Keyed by joint
    name in YAML iteration order.  Both ``sim`` and ``real`` are required for
    hardware commands; ``sim`` alone is accepted for sim-only tooling."""

    def joint_bounds(self) -> tuple[list[str], list[float], list[float]]:
        """Return ``(joint_names, sim_lower_rad, sim_upper_rad)`` in YAML order."""
        names: list[str] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for name, entry in self.joint_limits.items():
            names.append(name)
            lowers.append(entry.lower_rad)
            uppers.append(entry.upper_rad)
        return names, lowers, uppers

    def with_sim_limits(
        self, sim_lower_rad: list[float], sim_upper_rad: list[float]
    ) -> "RobotConfig":
        """Return a copy of this config with ``sim`` bounds filled from a bundle.

        The bundle's ``joint_lower_rad`` / ``joint_upper_rad`` are in the same
        iteration order as ``self.joint_limits``.  Any joint that already has
        ``sim`` bounds set is overwritten.
        """
        import dataclasses

        joint_names = list(self.joint_limits.keys())
        if len(sim_lower_rad) != len(joint_names) or len(sim_upper_rad) != len(joint_names):
            raise ValueError(
                f"with_sim_limits: expected {len(joint_names)} bounds "
                f"(one per joint in joint_limits), got "
                f"{len(sim_lower_rad)} lower and {len(sim_upper_rad)} upper."
            )
        new_limits = {}
        for i, name in enumerate(joint_names):
            entry = self.joint_limits[name]
            new_limits[name] = dataclasses.replace(
                entry,
                sim_lower_rad=sim_lower_rad[i],
                sim_upper_rad=sim_upper_rad[i],
            )
        return dataclasses.replace(self, joint_limits=new_limits)

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
        if "joint_calibration" in data:
            raise ValueError(
                f"Robot config uses the removed 'joint_calibration' section.\n"
                f"Migrate to the unified 'joint_limits' schema:\n\n"
                f"  joint_limits:\n"
                f"    shoulder_pan:\n"
                f"      sim:  [<lower_rad>, <upper_rad>]   # URDF canonical radians\n"
                f"      real: [<lower_rad>, <upper_rad>]   # LeRobot motor radians at stops\n"
                f"    ...\n\n"
                f"Run 'python -m so101_real.joint_calibrate --mode sweep' to re-measure\n"
                f"'real' bounds, then update robot.yaml.\n"
                f"Config path: {path}"
            )

        robot = data.get("robot")
        if robot is None:
            raise ValueError(
                f"Robot config YAML must contain a top-level 'robot' key: {path}"
            )
        required = {"port", "calibration_file"}
        missing = required - set(robot)
        if missing:
            raise ValueError(
                f"Robot config is missing required keys: {sorted(missing)}\n"
                f"Config path: {path}"
            )

        # Parse joint_limits (required) — must use the unified sim/real schema.
        jl_raw = data.get("joint_limits")
        if not jl_raw:
            raise ValueError(
                f"robot.yaml is missing a required 'joint_limits' section.\n"
                f"Config path: {path}"
            )
        joint_limits: dict[str, JointLimitEntry] = {}
        for jname, jentry in jl_raw.items():
            # Detect old anonymous-list / lower_rad+upper_rad schema.
            if isinstance(jentry, dict) and (
                "lower_rad" in jentry or "upper_rad" in jentry
            ):
                raise ValueError(
                    f"joint_limits[{jname!r}] uses the old 'lower_rad'/'upper_rad' "
                    f"schema.  Migrate to the unified sim/real schema:\n\n"
                    f"  {jname}:\n"
                    f"    sim:  [<lower_rad>, <upper_rad>]\n"
                    f"    real: [<lower_rad>, <upper_rad>]\n\n"
                    f"Config path: {path}"
                )
            sim_lo: Optional[float] = None
            sim_hi: Optional[float] = None
            real_lo: Optional[float] = None
            real_hi: Optional[float] = None
            if isinstance(jentry, dict):
                sim_raw = jentry.get("sim")
                real_raw = jentry.get("real")
                if sim_raw is not None:
                    if not (isinstance(sim_raw, (list, tuple)) and len(sim_raw) == 2):
                        raise ValueError(
                            f"joint_limits[{jname!r}].sim must be a two-element list "
                            f"[lower_rad, upper_rad]; got {sim_raw!r}. Config path: {path}"
                        )
                    sim_lo, sim_hi = float(sim_raw[0]), float(sim_raw[1])
                    if sim_hi <= sim_lo:
                        raise ValueError(
                            f"joint_limits[{jname!r}].sim[1] must be > sim[0]; "
                            f"got [{sim_lo}, {sim_hi}]. Config path: {path}"
                        )
                if real_raw is not None:
                    if not (isinstance(real_raw, (list, tuple)) and len(real_raw) == 2):
                        raise ValueError(
                            f"joint_limits[{jname!r}].real must be a two-element list "
                            f"[lower_rad, upper_rad]; got {real_raw!r}. Config path: {path}"
                        )
                    real_lo, real_hi = float(real_raw[0]), float(real_raw[1])
            joint_limits[str(jname)] = JointLimitEntry(
                sim_lower_rad=sim_lo,
                sim_upper_rad=sim_hi,
                real_lower_rad=real_lo,
                real_upper_rad=real_hi,
            )

        reset_pose: Optional[ResetPoseCfg] = None
        rp_data = data.get("reset_pose")
        if rp_data is not None:
            if "joints_rad" in rp_data:
                raise ValueError(
                    f"reset_pose uses the legacy key 'joints_rad'.  Replace it with "
                    f"'joints' and add 'unit: rad' (or 'deg' / 'norm'). "
                    f"Config path: {path}"
                )
            required_rp = {"enabled", "joints", "unit", "duration_s"}
            missing_rp = required_rp - set(rp_data)
            if missing_rp:
                raise ValueError(
                    f"reset_pose config is missing required keys: {sorted(missing_rp)}\n"
                    f"Config path: {path}"
                )
            _valid_units = ("rad", "deg", "norm")
            _unit = str(rp_data["unit"])
            if _unit not in _valid_units:
                raise ValueError(
                    f"reset_pose.unit must be one of {_valid_units}; got {_unit!r}. "
                    f"Config path: {path}"
                )
            _joints_raw = [float(v) for v in rp_data["joints"]]
            if _unit == "rad":
                _joints_rad = _joints_raw
            elif _unit == "deg":
                _joints_rad = [math.radians(v) for v in _joints_raw]
            else:  # norm — requires joint_limits
                _jnames = list(joint_limits.keys())
                _lowers = [joint_limits[n].lower_rad for n in _jnames]
                _uppers = [joint_limits[n].upper_rad for n in _jnames]
                _conv = from_robot_config(_jnames, lower_rad=_lowers, upper_rad=_uppers)
                _joints_rad = _conv.to_canonical_rad(_joints_raw, "norm").tolist()
            reset_pose = ResetPoseCfg(
                enabled=bool(rp_data["enabled"]),
                joints_rad=_joints_rad,
                duration_s=float(rp_data["duration_s"]),
            )

        max_delta_raw = robot.get("max_delta_rad")
        max_delta_rad: Optional[float] = (
            float(max_delta_raw) if max_delta_raw is not None else None
        )
        return cls(
            port=str(robot["port"]),
            calibration_file=str(robot["calibration_file"]),
            max_delta_rad=max_delta_rad,
            reset_pose=reset_pose,
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

        # Validate that all requested joints have 'real' bounds — required to
        # derive the LeRobot calibration transform.
        missing_real = [
            n for n in joint_names
            if n not in config.joint_limits or not config.joint_limits[n].has_real
        ]
        if missing_real:
            raise ValueError(
                f"So101Robot: joint_limits is missing 'real' bounds for: {missing_real}.\n"
                f"Run 'python -m so101_real.joint_calibrate --mode sweep' and add\n"
                f"'real: [lower_rad, upper_rad]' under each joint in robot.yaml."
            )

        # Build per-joint lero_scale / lero_offset_rad from the unified limits.
        lero_scale = [config.joint_limits[n].lero_scale for n in joint_names]
        lero_offset = [config.joint_limits[n].lero_offset_rad for n in joint_names]

        # Single source of truth for canonical ↔ LeRobot conversions.
        # All rad/deg/lrad/ldeg arithmetic lives in JointUnitConverter; this
        # class only knows about reading degrees off the bus and writing
        # degrees back to it.
        self._units = from_robot_config(
            joint_names=joint_names,
            lero_scale=lero_scale,
            lero_offset_rad=lero_offset,
        )

        print(
            f"[So101Robot] Calibration loaded for: {joint_names}"
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

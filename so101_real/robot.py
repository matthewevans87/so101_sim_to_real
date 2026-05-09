"""robot.py — SO-101 robot interface for real-robot inference.

Wraps the LeRobot SO101Follower API.  All external communication uses
degrees (LeRobot convention).  Internally we work in radians to match
the training obs/action contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml


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
        return cls(
            port=str(robot["port"]),
            calibration_file=str(robot["calibration_file"]),
            max_delta_rad=float(robot["max_delta_rad"]),
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

        Returns
        -------
        torch.Tensor
            Shape ``(n_joints,)`` float32, values in **radians**.
        """
        if self._follower is None:
            # dry_run mode — return zeros
            return torch.zeros(len(self._joint_names))

        obs = self._follower.get_observation()
        positions_deg = [obs[f"{name}.pos"] for name in self._joint_names]
        positions_rad = [math.radians(d) for d in positions_deg]
        return torch.tensor(positions_rad, dtype=torch.float32)

    def send_joints(self, q_target_rad: torch.Tensor) -> None:
        """Send joint position targets.

        Parameters
        ----------
        q_target_rad:
            Shape ``(n_joints,)`` float32, values in **radians**.
        """
        if self._follower is None:
            return  # dry_run mode

        q_np = q_target_rad.detach().cpu().float()
        action = {
            f"{name}.pos": math.degrees(float(q_np[i]))
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

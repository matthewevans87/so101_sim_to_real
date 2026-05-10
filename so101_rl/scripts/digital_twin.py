"""digital_twin.py — Isaac Sim digital twin viewer for SO-101 real-robot deployment.

Subscribes to ``/so101/joint_states`` (``sensor_msgs/JointState``) published by
the deploy loop (``run.py deploy --ros``) and mirrors the positions to the SO-101
articulation in a live Isaac Sim viewport.  Run this in a separate terminal
alongside ``run.py deploy --ros``.

Usage
-----
# With run.py (handles env vars and display automatically):
    ./scripts/run.py digital-twin

# Or manually via Isaac Lab:
    isaaclab.sh -p so101_rl/scripts/digital_twin.py [--display N]

Requires
--------
- Isaac Lab / Isaac Sim 5.1.0 (run via isaaclab.sh)
- ROS2 Jazzy sourced in the calling environment (or injected by run.py)
- ``run.py deploy --ros ...`` running in a separate terminal
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing (MUST happen before AppLauncher)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="SO-101 digital twin viewer — mirrors real robot joints in Isaac Sim.",
)
parser.add_argument(
    "--topic",
    default="/so101/joint_states",
    metavar="TOPIC",
    help="ROS2 JointState topic to subscribe to (default: /so101/joint_states)",
)
parser.add_argument(
    "--display",
    type=int,
    default=None,
    dest="display_sock",
    metavar="N",
    help="X11 display socket number (sets DISPLAY=:N). Auto-discovered if omitted.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# X11 / display setup — must happen BEFORE AppLauncher
# ---------------------------------------------------------------------------
import glob
import os
import re


def _resolve_x11(display_sock: int | None = None) -> None:
    if display_sock is not None:
        os.environ["DISPLAY"] = f":{display_sock}"

    disp = os.environ.get("DISPLAY", "")
    xauth = os.environ.get("XAUTHORITY", "")

    if not disp or not xauth:
        _gui_procs = re.compile(r"gnome-shell|plasmashell|xfce4-session|Xorg|Xwayland")
        for env_file in sorted(glob.glob("/proc/*/environ")):
            try:
                pid = env_file.split("/")[2]
                comm = ""
                try:
                    comm = open(f"/proc/{pid}/comm").read().strip()
                except OSError:
                    pass
                cmdline = ""
                try:
                    cmdline = (
                        open(f"/proc/{pid}/cmdline", "rb")
                        .read()
                        .replace(b"\x00", b" ")
                        .decode("utf-8", errors="replace")
                    )
                except OSError:
                    pass
                if not _gui_procs.search(comm + " " + cmdline):
                    continue
                raw = open(env_file, "rb").read()
                env_vars: dict[str, str] = {}
                for entry in raw.split(b"\x00"):
                    if b"=" in entry:
                        k, _, v = entry.partition(b"=")
                        env_vars[k.decode("utf-8", errors="replace")] = v.decode(
                            "utf-8", errors="replace"
                        )
                if not disp and env_vars.get("DISPLAY"):
                    os.environ["DISPLAY"] = env_vars["DISPLAY"]
                    disp = env_vars["DISPLAY"]
                if not xauth and env_vars.get("XAUTHORITY"):
                    os.environ["XAUTHORITY"] = env_vars["XAUTHORITY"]
                    xauth = env_vars["XAUTHORITY"]
                if disp and xauth:
                    break
            except (OSError, ValueError, IndexError):
                continue

    if "XAUTHORITY" not in os.environ:
        from pathlib import Path as _Path

        user = os.environ.get("USER", "")
        for candidate in [
            _Path.home() / ".Xauthority",
            _Path(f"/home/{user}/.Xauthority"),
        ]:
            if candidate.is_file():
                os.environ["XAUTHORITY"] = str(candidate)
                break


_resolve_x11(args_cli.display_sock)

# headless=False so the viewport is visible
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All other imports come AFTER AppLauncher
# ---------------------------------------------------------------------------

import json
import subprocess
import sys
import threading
from pathlib import Path

import torch
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

# Project-local imports (available because so101_rl is installed in env_isaaclab)
from so101_rl.configurations.so101 import SO101_CFG
from so101_rl.configurations.table import TABLE_CFG

# Joint names must match the publisher side (so101_real bundle active_joints order)
_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ---------------------------------------------------------------------------
# ROS2 joint state listener — runs system Python in a subprocess
# ---------------------------------------------------------------------------
# Isaac Sim embeds Python 3.12 but the bundled rclpy .so is built for 3.11,
# so rclpy cannot be imported inside the Isaac Sim process.  Instead we spawn
# a lightweight subprocess using the system Python (where rclpy is fully
# functional) that subscribes to the ROS2 topic and streams joint states as
# newline-delimited JSON to stdout.  A daemon thread in the main process reads
# that stream and stores the latest positions for the sim loop.

_LISTENER_SCRIPT = """\
import sys, json, os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

rclpy.init()
node = Node("dt_listener")

def cb(msg):
    if msg.position:
        sys.stdout.write(json.dumps({"names": list(msg.name), "positions": list(msg.position)}) + "\\n")
        sys.stdout.flush()

node.create_subscription(JointState, sys.argv[1], cb, 10)
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
"""


class _JointStateListener:
    """Spawns a rclpy subprocess and exposes the latest joint positions."""

    # Python 3.12 is required — it matches the ABI of the system rclpy .so.
    # The default python3 on this machine is 3.14 (linuxbrew) which is incompatible.
    _PYTHON = "/usr/bin/python3.12"

    def __init__(self, topic: str, joint_names: list[str]) -> None:
        self._joint_names = joint_names
        self._positions: list[float] | None = None
        self._lock = threading.Lock()

        # Write the listener script to a temp file to avoid shell quoting issues
        # with the -c approach (curly braces in the JSON call get mangled).
        import tempfile
        self._tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="dt_listener_"
        )
        self._tmpfile.write(_LISTENER_SCRIPT)
        self._tmpfile.flush()
        self._tmpfile.close()

        env = {
            "ROS_DISTRO": "jazzy",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "PYTHONPATH": "/opt/ros/jazzy/lib/python3.12/site-packages",
            "LD_LIBRARY_PATH": "/opt/ros/jazzy/lib",
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
        }

        self._proc = subprocess.Popen(
            [self._PYTHON, self._tmpfile.name, topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        print(f"[digital_twin] ROS2 listener started (pid={self._proc.pid}, topic={topic})")

    def _stderr_loop(self) -> None:
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                print(f"[dt_listener] {line}")

    def _read_loop(self) -> None:
        for line in self._proc.stdout:
            try:
                data = json.loads(line)
                names: list[str] = data["names"]
                positions: list[float] = data["positions"]
                # Re-order to match self._joint_names
                name_to_pos = dict(zip(names, positions))
                ordered = [name_to_pos.get(n) for n in self._joint_names]
                if all(v is not None for v in ordered):
                    with self._lock:
                        self._positions = ordered  # type: ignore[assignment]
            except (json.JSONDecodeError, KeyError):
                pass

    def get_positions(self) -> list[float] | None:
        with self._lock:
            return self._positions

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        Path(self._tmpfile.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Scene — robot + table, no camera sensor, no cube
# ---------------------------------------------------------------------------


@configclass
class DigitalTwinSceneCfg(InteractiveSceneCfg):
    """Minimal scene for joint mirroring: SO-101 robot + table."""

    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
    table = TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")  # type: ignore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Simulation setup ──────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 1.2, 0.8], target=[0.3, 0.0, 0.4])

    # Dome light for even, bright illumination — no ground plane (table suffices)
    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

    scene_cfg = DigitalTwinSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]

    # ── Resolve joint indices from _JOINT_NAMES ────────────────────────────────
    # IsaacLab reports joint names via robot.data.joint_names (alphabetical order).
    # We need to map our topic's name order to the articulation's DOF indices.
    sim_joint_names: list[str] = robot.data.joint_names
    try:
        joint_indices = [sim_joint_names.index(n) for n in _JOINT_NAMES]
    except ValueError as exc:
        print(f"[digital_twin] ERROR: joint name mismatch — {exc}", file=sys.stderr)
        print(f"  Sim joints : {sim_joint_names}", file=sys.stderr)
        print(f"  Expected   : {_JOINT_NAMES}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    print(f"[digital_twin] Sim joint order: {sim_joint_names}")
    print(f"[digital_twin] Mapped indices : {joint_indices}")

    # ── Start ROS2 listener subprocess ───────────────────────────────────────
    listener = _JointStateListener(topic=args_cli.topic, joint_names=_JOINT_NAMES)

    print("[digital_twin] Scene ready.")
    print("[digital_twin] Running. Close the viewport window or press Ctrl-C to quit.")

    # ── Simulation loop ───────────────────────────────────────────────────────
    try:
        while simulation_app.is_running():
            positions = listener.get_positions()
            if positions is not None:
                # Build a (1, num_dof) position tensor in the articulation's DOF order.
                pos_tensor = torch.zeros(1, robot.num_joints, device=robot.device)
                for src_idx, dst_idx in enumerate(joint_indices):
                    pos_tensor[0, dst_idx] = positions[src_idx]
                robot.set_joint_position_target(pos_tensor)
                robot.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
    except KeyboardInterrupt:
        print("\n[digital_twin] Interrupted.")
    finally:
        listener.close()

    simulation_app.close()


if __name__ == "__main__":
    main()

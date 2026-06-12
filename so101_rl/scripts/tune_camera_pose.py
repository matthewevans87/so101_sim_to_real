"""tune_camera_pose.py — Interactive wrist-camera extrinsic tuning tool.

Spawns the SO-101 robot in a minimal Isaac Lab scene, mirrors live joint
positions from the physical robot (or uses a fixed pose with --no-robot),
renders the wrist camera, and shows a continuously-updated OpenCV overlay
blending the sim render with a pre-captured real frame.

Use the Isaac Sim viewport gizmos to drag the CameraXframe prim until the
overlay converges.  Press ``s`` in the OpenCV window to print the final
CameraXframe transform as Python literals, USD xformOp values, and YAML.

Usage
-----
# With real robot connected:
isaaclab.sh -p so101_rl/scripts/tune_camera_pose.py \
    --robot-config so101_real/configs/robot.yaml

# Offline / no robot:
isaaclab.sh -p so101_rl/scripts/tune_camera_pose.py --no-robot

# Custom real frame and initial display width:
isaaclab.sh -p so101_rl/scripts/tune_camera_pose.py \
    --real-image so101_real/calibration/captures/live_frame.png \
    --display-width 1600 \
    --no-robot

Hotkeys (in the OpenCV window)
--------------------------------
  [ / ]   decrease / increase blend alpha (step 0.05)
  c       cycle view modes: blend → side-by-side → checkerboard → abs-diff → blend
  s       snapshot CameraXframe transform to stdout (+ --out-yaml if given)
  r       reload real image from disk
  q       quit
"""

# ---------------------------------------------------------------------------
# IMPORTANT: Isaac Sim must be launched BEFORE any other imports.
# All non-standard imports must be deferred until after AppLauncher.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing (must happen before AppLauncher)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Interactive wrist-camera extrinsic tuning tool for SO-101.",
)
parser.add_argument(
    "--real-image",
    default="so101_real/calibration/captures/live_frame.png",
    dest="real_image",
    metavar="PATH",
    help="Path to the reference real-camera frame (default: so101_real/calibration/captures/live_frame.png)",
)
parser.add_argument(
    "--robot-config",
    default=None,
    dest="robot_config",
    metavar="PATH",
    help="Path to so101_real robot config YAML. Required unless --no-robot.",
)
parser.add_argument(
    "--no-robot",
    action="store_true",
    dest="no_robot",
    help="Skip real-robot connection; use --joint-pose for a fixed sim pose.",
)
parser.add_argument(
    "--joint-pose",
    default=None,
    dest="joint_pose",
    metavar="YAML",
    help="YAML file with joint_name: value_rad entries for --no-robot mode. "
    "Defaults to all-zeros if omitted.",
)
parser.add_argument(
    "--display-width",
    type=int,
    default=1280,
    dest="display_width",
    metavar="PX",
    help="Maximum width of the OpenCV display window in pixels (default: 1280).",
)
parser.add_argument(
    "--out-yaml",
    default=None,
    dest="out_yaml",
    metavar="PATH",
    help="Write final CameraXframe transform YAML to this file on exit/snapshot.",
)
parser.add_argument(
    "--render-width",
    type=int,
    default=640,
    dest="render_width",
    metavar="PX",
    help="Width of the sim camera render in pixels (default: 640).",
)
parser.add_argument(
    "--render-height",
    type=int,
    default=360,
    dest="render_height",
    metavar="PX",
    help="Height of the sim camera render in pixels (default: 360).",
)
parser.add_argument(
    "--display",
    type=int,
    default=None,
    dest="display_sock",
    metavar="N",
    help="X11 display socket number (sets DISPLAY=:N before launching Isaac Sim). "
    "Defaults to auto-discovery from running desktop processes.",
)
parser.add_argument(
    "--lerobot-python",
    default=None,
    dest="lerobot_python",
    metavar="PATH",
    help="Path to the Python interpreter in the lerobot conda env. "
    "Defaults to ~/.conda/envs/lerobot/bin/python.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# X11 / display resolution  (mirrors the logic in scripts/run.py)
# Must run BEFORE AppLauncher so Isaac Sim picks up the right DISPLAY.
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
        # Scan running desktop processes for DISPLAY/XAUTHORITY
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

    # Last-resort XAUTHORITY fallback
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

    print(f"[tune] DISPLAY={os.environ.get('DISPLAY', '(not set)')}")
    print(f"[tune] XAUTHORITY={os.environ.get('XAUTHORITY', '(not set)')}")


_resolve_x11(args_cli.display_sock)

# Enable camera rendering — required for TiledCamera / Camera sensors.
args_cli.enable_cameras = True

# headless=False so the Isaac Sim viewport is visible for gizmo manipulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All other imports come AFTER AppLauncher
# ---------------------------------------------------------------------------

import math
import sys
import time
from pathlib import Path

import numpy as np
import subprocess
import json
import termios
import tty
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from pxr import Gf, UsdGeom

# Ensure ISAAC_LAB_WORKSPACE_PATH points to the project root.
_project_root = Path(__file__).resolve().parents[2]
if "ISAAC_LAB_WORKSPACE_PATH" not in os.environ:
    os.environ["ISAAC_LAB_WORKSPACE_PATH"] = str(_project_root)

# Project-local imports (available after AppLauncher because so101_rl is installed)
from so101_rl.configurations.so101 import SO101_CFG
from so101_rl.configurations.camera import build_gripper_tiled_camera_cfg
from so101_rl.helpers.opencv_to_isaac_camera import load_intrinsics
from so101_rl.configurations.so101_env_params import So101EnvParams

# cv2 is available in this env but only for image I/O — imshow/waitKey do not work
# because Isaac Sim ships headless OpenCV (no GTK).  Display is handled via ffplay.
import cv2

# Load intrinsics from the env YAML (SO101_ENV_CONFIG must be set by the caller).
_env_config_path = os.environ.get("SO101_ENV_CONFIG")
if not _env_config_path:
    raise RuntimeError(
        "SO101_ENV_CONFIG is not set. "
        "Pass --env_config to run.py or export SO101_ENV_CONFIG before running this script."
    )
_tune_env_params = So101EnvParams.load(_env_config_path)
_tune_cam_sensor = _tune_env_params.sensors.camera
_tune_intrinsics = load_intrinsics(
    Path(os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace"))
    / _tune_cam_sensor.intrinsics_path
)
_TILED_CAMERA_CFG = build_gripper_tiled_camera_cfg(
    _tune_intrinsics, args_cli.render_height, args_cli.render_width
)

# ---------------------------------------------------------------------------
# Joint name order expected by the sim articulation and the real robot
# ---------------------------------------------------------------------------
_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------


@configclass
class TuneCameraSceneCfg(InteractiveSceneCfg):
    """Minimal scene: SO-101 + wrist camera. Ground is spawned separately."""

    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    camera: TiledCameraCfg = _TILED_CAMERA_CFG.replace(  # type: ignore
        data_types=["rgb"],
    )


# ---------------------------------------------------------------------------
# USD transform helpers
# ---------------------------------------------------------------------------


def _read_xframe_transform(stage, prim_path: str) -> tuple[tuple, tuple] | None:
    """Read (translate, orient_wxyz) from a USD XForm prim.

    Returns None if the prim is invalid or has no xformOps.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[tune] WARNING: prim not found: {prim_path}")
        return None

    xformable = UsdGeom.Xformable(prim)
    translate = None
    orient = None

    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate and translate is None:
            v = op.Get()
            if v is not None:
                translate = (float(v[0]), float(v[1]), float(v[2]))
        elif op_type == UsdGeom.XformOp.TypeOrient and orient is None:
            q = op.Get()
            if q is not None:
                # Gf.Quatd / Quatf: .GetReal() = w, .GetImaginary() = (x, y, z)
                orient = (
                    float(q.GetReal()),
                    float(q.GetImaginary()[0]),
                    float(q.GetImaginary()[1]),
                    float(q.GetImaginary()[2]),
                )

    if translate is None:
        translate = (0.0, 0.0, 0.0)
    if orient is None:
        orient = (1.0, 0.0, 0.0, 0.0)

    return translate, orient


def _print_transform(
    translate: tuple, orient_wxyz: tuple, out_yaml_path: str | None = None
) -> None:
    """Print transform in three formats and optionally write YAML."""
    tx, ty, tz = translate
    w, x, y, z = orient_wxyz

    print("\n" + "=" * 60)
    print("CameraXframe transform snapshot")
    print("=" * 60)

    print("\n--- Python literals (paste into camera.py) ---")
    print(f"CAMERA_TRANSLATE_VEC = ({tx}, {ty}, {tz})")
    print(f"CAMERA_ROTATION_QUAT_WXYZ = ({w}, {x}, {y}, {z})")

    print("\n--- USD xformOp values ---")
    print(f"xformOp:translate  =  ({tx:.8f},  {ty:.8f},  {tz:.8f})")
    print(f"xformOp:orient     =  w={w:.8f}  x={x:.8f}  y={y:.8f}  z={z:.8f}")

    payload = {
        "camera_xframe_transform": {
            "translate": {"x": tx, "y": ty, "z": tz},
            "orient_wxyz": {"w": w, "x": x, "y": y, "z": z},
        }
    }
    print("\n--- YAML ---")
    print(yaml.dump(payload, default_flow_style=False, sort_keys=False))

    if out_yaml_path:
        p = Path(out_yaml_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        print(f"[tune] Written to: {p}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Overlay rendering helpers
# ---------------------------------------------------------------------------

_VIEW_MODES = ["blend", "side-by-side", "checkerboard", "abs-diff"]


def _build_composite(
    real_bgr: np.ndarray,
    sim_bgr: np.ndarray,
    alpha: float,
    mode: str,
    checker_size: int = 64,
) -> np.ndarray:
    H, W = real_bgr.shape[:2]

    if mode == "blend":
        return cv2.addWeighted(real_bgr, 1.0 - alpha, sim_bgr, alpha, 0.0)

    if mode == "side-by-side":

        def _label(img, text):
            out = img.copy()
            cv2.putText(
                out,
                text,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return out

        return np.concatenate(
            [_label(real_bgr, "real"), _label(sim_bgr, "sim")], axis=1
        )

    if mode == "checkerboard":
        out = real_bgr.copy()
        n = checker_size
        for row in range(0, H, 2 * n):
            for col in range(0, W, 2 * n):
                out[row : row + n, col : col + n] = sim_bgr[
                    row : row + n, col : col + n
                ]
                r2 = min(row + 2 * n, H)
                c2 = min(col + 2 * n, W)
                out[row + n : r2, col + n : c2] = sim_bgr[row + n : r2, col + n : c2]
        return out

    if mode == "abs-diff":
        diff = cv2.absdiff(real_bgr, sim_bgr)
        diff_amplified = cv2.convertScaleAbs(diff, alpha=3.0)
        return cv2.applyColorMap(diff_amplified, cv2.COLORMAP_INFERNO)

    return real_bgr  # fallback


def _load_real_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(f"[tune] ERROR: could not read real image: {path}")
    return img


# ---------------------------------------------------------------------------
# ffplay pipe display
# ---------------------------------------------------------------------------
# Isaac Sim's bundled OpenCV has no GTK/Qt so cv2.imshow raises an error.
# Instead we open ffplay as a subprocess and push raw RGB24 frames to its
# stdin pipe.  ffplay opens its own X11 window and handles the display.


class FfplayDisplay:
    """Write numpy RGB frames to an ffplay window via stdin pipe."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "tune-camera-pose",
        display_width: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        # Scale display if requested
        vf = ""
        if display_width and display_width < width:
            scaled_h = int(height * display_width / width)
            # Ensure even dimensions (required by most codecs)
            scaled_h = scaled_h + (scaled_h % 2)
            display_w = display_width + (display_width % 2)
            vf = f"-vf scale={display_w}:{scaled_h}"
        vf_args = vf.split() if vf else []
        cmd = [
            "ffplay",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-window_title",
            title,
            "-an",  # no audio
            "-autoexit",  # close when stdin closes
            *vf_args,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_rgb: np.ndarray) -> bool:
        """Push one RGB frame.  Returns False if ffplay has exited."""
        if self._proc.poll() is not None:
            return False
        try:
            self._proc.stdin.write(frame_rgb.tobytes())
            self._proc.stdin.flush()
        except BrokenPipeError:
            return False
        return True

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Non-blocking terminal keyboard (replaces cv2.waitKey)
# ---------------------------------------------------------------------------


class _RawTerminal:
    """Context manager: put stdin in raw mode for single-char reads."""

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read_key(self) -> str | None:
        """Return a character if one is waiting, else None (non-blocking)."""
        import select

        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
# Real-robot interface — subprocess bridge
# ---------------------------------------------------------------------------
# lerobot is only installed in the 'lerobot' conda env.  We spawn robot_bridge.py
# as a subprocess using that interpreter so env_isaaclab stays isolated.


class RobotBridgeReader:
    """Spawn robot_bridge.py in the lerobot env; read joint positions via pipe."""

    def __init__(
        self, robot_config_path: str, lerobot_python: str | None = None
    ) -> None:
        import collections
        import threading

        if lerobot_python is None:
            # lerobot is pip-installed into the miniforge base env's Python 3.12.
            # Check common locations in order.
            _candidates = [
                "/opt/miniforge3/bin/python3.12",
                "/opt/miniforge3/bin/python3",
                str(Path.home() / ".conda" / "envs" / "lerobot" / "bin" / "python"),
                "/opt/conda/bin/python3.12",
            ]
            lerobot_python = next(
                (p for p in _candidates if Path(p).exists()), _candidates[0]
            )

        bridge_script = Path(__file__).resolve().parent / "robot_bridge.py"
        if not bridge_script.exists():
            raise FileNotFoundError(f"robot_bridge.py not found at {bridge_script}")
        if not Path(lerobot_python).exists():
            raise FileNotFoundError(
                f"lerobot Python not found: {lerobot_python}\n"
                "Pass --lerobot-python to override."
            )

        self._latest: dict[str, float] = {name: 0.0 for name in _JOINT_NAMES}
        self._lock = threading.Lock()
        self._running = True

        # Build a clean environment for the bridge subprocess so Isaac Sim's
        # PYTHONPATH (which contains its own torch build) does not shadow the
        # lerobot conda env's packages.  Keep only basic OS vars.
        _PASSTHROUGH = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TERM",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "DISPLAY",
            "XAUTHORITY",
            "LD_LIBRARY_PATH",  # keep for system libs (e.g. libusb for serial)
        }
        bridge_env = {k: v for k, v in os.environ.items() if k in _PASSTHROUGH}
        # Explicitly clear variables that Isaac Sim injects
        for _var in (
            "PYTHONPATH",
            "PYTHONHOME",
            "ISAAC_LAB_WORKSPACE_PATH",
            "CARB_APP_PATH",
            "EXP_PATH",
            "OMNI_KIT_ALLOW_ROOT",
        ):
            bridge_env.pop(_var, None)

        self._proc = subprocess.Popen(
            [lerobot_python, str(bridge_script), "--robot-config", robot_config_path],
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # bridge errors visible in this terminal
            text=True,
            bufsize=1,
            env=bridge_env,
        )

        # Wait for the "ready" signal — skip any non-JSON lines (e.g. robot
        # connect messages printed by so101_real to stdout).
        first_line = ""
        msg = {}
        while True:
            line = self._proc.stdout.readline()
            if not line:  # EOF — bridge exited
                self._proc.wait()
                raise RuntimeError(
                    f"robot_bridge.py exited before sending 'ready' (exit code "
                    f"{self._proc.returncode}). Check stderr above for details."
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                first_line = line
                break
            except json.JSONDecodeError:
                # Non-JSON line from robot connect — forward to our stderr so
                # the user can still see it, then keep waiting.
                print(f"[bridge] {line}", file=sys.stderr, flush=True)
                continue
        if msg.get("status") != "ready":
            raise RuntimeError(f"Unexpected bridge first message: {first_line!r}")
        print(f"[tune] Robot bridge ready (pid {self._proc.pid})")

        # Background reader thread
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        for line in self._proc.stdout:
            if not self._running:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                with self._lock:
                    self._latest = data
            except json.JSONDecodeError:
                pass

    def read_joints(self) -> list[float]:
        """Return the latest joint positions in _JOINT_NAMES order."""
        with self._lock:
            return [self._latest.get(name, 0.0) for name in _JOINT_NAMES]

    def disconnect(self) -> None:
        self._running = False
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        print("[tune] Robot bridge stopped.")


def _load_joint_pose_yaml(path: str | None) -> dict[str, float]:
    """Load {joint_name: rad} from YAML; return zeros if path is None."""
    if path is None:
        return {name: 0.0 for name in _JOINT_NAMES}
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return {name: float(data.get(name, 0.0)) for name in _JOINT_NAMES}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate args
    if not args_cli.no_robot and args_cli.robot_config is None:
        print("ERROR: --robot-config is required unless --no-robot is set.")
        print("       Run with --help for usage.")
        sys.exit(1)

    real_image_path = args_cli.real_image

    # ── Simulation setup ────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.3, 0.0, 0.3])

    # Spawn ground plane before building the scene
    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.3, 0.3, 0.3))
    ground_cfg.func("/World/ground", ground_cfg)

    scene_cfg = TuneCameraSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[tune] Scene ready.")

    robot: Articulation = scene["robot"]
    camera: TiledCamera = scene["camera"]

    # Resolve joint indices in the articulation
    joint_indices, _ = robot.find_joints(_JOINT_NAMES)
    num_joints = len(joint_indices)

    # ── Real-robot / fixed-pose setup ────────────────────────────────────────
    real_robot = None
    fixed_joint_pos: torch.Tensor | None = None

    # Determine the sim device so tensors are placed correctly
    sim_device = sim.device

    if args_cli.no_robot:
        pose_dict = _load_joint_pose_yaml(args_cli.joint_pose)
        fixed_joint_pos = torch.zeros(
            (1, num_joints), dtype=torch.float32, device=sim_device
        )
        for i, name in enumerate(_JOINT_NAMES):
            fixed_joint_pos[0, i] = pose_dict[name]
        print(f"[tune] --no-robot mode. Fixed joint pose: {pose_dict}")
    else:
        real_robot = RobotBridgeReader(
            args_cli.robot_config,
            lerobot_python=args_cli.lerobot_python,
        )

    # ── Load real image ──────────────────────────────────────────────────────
    real_bgr = _load_real_image(real_image_path)
    if real_bgr is None:
        print(
            f"[tune] WARNING: real image not found at {real_image_path}. "
            "Using grey placeholder."
        )
        real_bgr = np.full(
            (args_cli.render_height, args_cli.render_width, 3), 128, dtype=np.uint8
        )

    # ── Overlay state ────────────────────────────────────────────────────────
    alpha = 0.5
    view_idx = 0  # index into _VIEW_MODES

    # ── USD stage (for reading CameraXframe transform) ───────────────────────
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    # InteractiveScene places env prims under /World/envs/env_N
    camera_xframe_path = (
        "/World/envs/env_0/Robot/gripper/mountscrew/camera_mount/CameraXframe"
    )
    # Verify the prim exists after reset (warn but continue if not — gizmo
    # path may differ until the first sim step initialises USD)
    _test_prim = stage.GetPrimAtPath(camera_xframe_path)
    if not _test_prim.IsValid():
        print(
            f"[tune] WARNING: CameraXframe prim not yet visible at {camera_xframe_path}"
        )
        print("[tune] It should appear after the first sim step.")

    # ── ffplay display ───────────────────────────────────────────────────────
    # Isaac Sim's OpenCV has no GTK; display via ffplay subprocess instead.
    display_w = args_cli.render_width
    display_h = args_cli.render_height
    ffplay = FfplayDisplay(
        width=display_w,
        height=display_h,
        title="tune-camera-pose (sim + real)",
        display_width=args_cli.display_width,
    )

    # ── Simulation loop ──────────────────────────────────────────────────────
    print(
        "[tune] Running. Hotkeys (in THIS terminal): [ ] alpha  c view  s snapshot  r reload  q quit"
    )
    print(f"[tune] CameraXframe prim: {camera_xframe_path}")

    with _RawTerminal() as kbd:
        try:
            while simulation_app.is_running():
                # -- Step sim
                sim.step()
                scene.update(sim.get_physics_dt())

                # -- Apply joint positions
                if real_robot is not None:
                    raw = real_robot.read_joints()  # list[float], _JOINT_NAMES order
                    q = torch.zeros(
                        (1, num_joints), dtype=torch.float32, device=sim_device
                    )
                    for i in range(num_joints):
                        q[0, i] = float(raw[i])
                    robot.write_joint_position_to_sim(q, joint_ids=joint_indices)
                    robot.write_joint_velocity_to_sim(
                        torch.zeros_like(q), joint_ids=joint_indices
                    )
                    robot.write_data_to_sim()
                else:
                    robot.write_joint_position_to_sim(
                        fixed_joint_pos, joint_ids=joint_indices
                    )
                    robot.write_joint_velocity_to_sim(
                        torch.zeros_like(fixed_joint_pos), joint_ids=joint_indices
                    )
                    robot.write_data_to_sim()

                # -- Get camera frame
                rgb_tensor = camera.data.output.get("rgb")
                if rgb_tensor is None or rgb_tensor.shape[0] == 0:
                    continue

                # rgb_tensor: (num_envs, H, W, 3) uint8 — TiledCamera pre-allocates correct shape
                sim_rgb = rgb_tensor[0].cpu().numpy()  # (H, W, 3) RGB uint8
                sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

                # -- Resize real image to sim render resolution for overlay
                rH, rW = sim_bgr.shape[:2]
                real_resized = cv2.resize(
                    real_bgr, (rW, rH), interpolation=cv2.INTER_AREA
                )

                # -- Build overlay
                composite_bgr = _build_composite(
                    real_resized, sim_bgr, alpha, _VIEW_MODES[view_idx]
                )

                # -- Add HUD (burn into frame)
                mode_label = _VIEW_MODES[view_idx]
                alpha_str = f" a={alpha:.2f}" if mode_label == "blend" else ""
                hud = f"[{mode_label}{alpha_str}]  [ ] alpha  c view  s snap  r reload  q quit"
                cv2.putText(
                    composite_bgr,
                    hud,
                    (8, composite_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    composite_bgr,
                    hud,
                    (8, composite_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                # -- Push to ffplay (BGR → RGB for pipe)
                composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
                alive = ffplay.write(composite_rgb)
                if not alive:
                    print("[tune] ffplay window closed — exiting.")
                    break

                # -- Keyboard (non-blocking, reads from THIS terminal)
                ch = kbd.read_key()
                if ch == "q":
                    break
                elif ch == "[":
                    alpha = max(0.0, alpha - 0.05)
                    print(f"\r[tune] alpha={alpha:.2f}  ", end="", flush=True)
                elif ch == "]":
                    alpha = min(1.0, alpha + 0.05)
                    print(f"\r[tune] alpha={alpha:.2f}  ", end="", flush=True)
                elif ch == "c":
                    view_idx = (view_idx + 1) % len(_VIEW_MODES)
                    print(
                        f"\r[tune] view={_VIEW_MODES[view_idx]}    ", end="", flush=True
                    )
                elif ch == "r":
                    new_img = _load_real_image(real_image_path)
                    if new_img is not None:
                        real_bgr = new_img
                        print(
                            f"\r[tune] Reloaded: {real_image_path}  ",
                            end="",
                            flush=True,
                        )
                elif ch == "s":
                    result = _read_xframe_transform(stage, camera_xframe_path)
                    if result is not None:
                        print()  # newline after \r status
                        _print_transform(*result, out_yaml_path=args_cli.out_yaml)

        except KeyboardInterrupt:
            pass
        finally:
            print("\n[tune] Exit — reading final CameraXframe transform...")
            result = _read_xframe_transform(stage, camera_xframe_path)
            if result is not None:
                _print_transform(*result, out_yaml_path=args_cli.out_yaml)

            ffplay.close()
            if real_robot is not None:
                real_robot.disconnect()
                print("[tune] Real robot disconnected.")


if __name__ == "__main__":
    main()
    simulation_app.close()

"""align_camera.py — Three-feed sim/real wrist-camera alignment + auto-fit tool.

Extends ``tune_camera_pose.py`` with:

* **Three video feeds** displayed simultaneously:
    1. real reference frame (from ffmpeg snapshot)
    2. sim camera (raw render — what Isaac Sim produces)
    3. sim camera *after the deploy image pipeline* — this is what the
       policy actually consumed during training/inference.
* **HUD diff metrics** between (real) and (sim-post-pipeline) at the
  policy's image resolution: mean absolute difference and Pearson NCC.
* **Optimizer hotkey ``o``** that runs an ORB-based Nelder-Mead fit of the
  CameraXframe mount transform against the real reference frame, writes
  the result back to the live USD prim, and prints the converged transform
  in the same format ``tune_camera_pose.py`` uses.

Usage
-----
    isaaclab.sh -p so101_rl/scripts/align_camera.py \\
        --robot-config so101_real/configs/robot.yaml \\
        --bundle /mnt/.../exports \\
        --real-image so101_real/calibration/captures/live_frame.png

Use ``--no-robot --joint-pose so101_real/configs/calibration_pose.yaml`` to
run offline against a fixed pose.

Hotkeys (THIS terminal)
-----------------------
  [ / ]   decrease / increase blend alpha (0.05 step)
  c       cycle comparison panel: blend → abs-diff → checkerboard → side-by-side
  o       run ORB + Nelder-Mead optimizer (writes new transform to USD live)
  d       print current diff metrics + transform to stdout
  s       snapshot CameraXframe transform (Python literals + YAML)
  r       reload real reference image from disk
  q       quit

Calibration target
------------------
The optimizer needs textured features. The recommended workflow is:

1. ``./scripts/run.py stream --robot-config ... --no-torque`` — pose the
   arm by hand to a known calibration pose, holding the cube in the
   gripper.
2. ``ffmpeg -f v4l2 ... live_frame.png`` — capture a single real frame.
3. Run this script with the same calibration joint pose YAML and the
   captured frame.

The black cube held in the gripper supplies a small textured target that is
sufficient for ORB given good lighting; if you see < 20 ORB matches at the
initial transform, the optimizer will refuse to run — add texture or improve
lighting and try again.
"""

# ---------------------------------------------------------------------------
# IMPORTANT: Isaac Sim must be launched BEFORE any other imports.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Three-feed sim/real wrist-camera alignment + auto-calibration.",
)
parser.add_argument(
    "--real-image",
    default="so101_real/calibration/captures/live_frame.png",
    dest="real_image",
    metavar="PATH",
    help="Reference real-camera frame (default: so101_real/calibration/captures/live_frame.png).",
)
parser.add_argument(
    "--bundle",
    default=None,
    dest="bundle",
    metavar="PATH",
    help="Deploy bundle directory (contains deploy_image_pipeline.yaml). "
    "Defaults to the scripts/pins/latest_bundle symlink.",
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
    help="YAML file with joint_name: value entries (used with --no-robot). "
    "Include a top-level 'unit:' key to specify the unit of the values: "
    "rad (default, canonical radians), deg (canonical degrees), "
    "norm (normalised [-1,1], requires --robot-config for joint_limits), "
    "lrad / ldeg (LeRobot radians/degrees, requires --robot-config for calibration).",
)
parser.add_argument(
    "--display-width",
    type=int,
    default=1600,
    dest="display_width",
    metavar="PX",
    help="Maximum width of the ffplay window (default: 1600).",
)
parser.add_argument(
    "--out-yaml",
    default=None,
    dest="out_yaml",
    metavar="PATH",
    help="Write final CameraXframe transform YAML to this file on snapshot/exit.",
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
    help="X11 display socket (DISPLAY=:N). Auto-detected from desktop if omitted.",
)
parser.add_argument(
    "--lerobot-python",
    default=None,
    dest="lerobot_python",
    metavar="PATH",
    help="Python interpreter for the lerobot env (defaults autodetected).",
)
parser.add_argument(
    "--optim-translation-bound-m",
    type=float,
    default=0.005,
    dest="optim_translation_bound_m",
    help="Optimizer per-axis translation bound in metres (default: 0.005).",
)
parser.add_argument(
    "--optim-rotation-bound-deg",
    type=float,
    default=5.0,
    dest="optim_rotation_bound_deg",
    help="Optimizer per-axis rotation bound in degrees (default: 5.0).",
)
parser.add_argument(
    "--optim-max-evals",
    type=int,
    default=80,
    dest="optim_max_evals",
    help="Optimizer max render evaluations (default: 80).",
)
parser.add_argument(
    "--optim-min-matches",
    type=int,
    default=20,
    dest="optim_min_matches",
    help="Minimum ORB matches required at the initial transform (default: 20).",
)
parser.add_argument(
    "--no-key-light",
    action="store_true",
    dest="no_key_light",
    help="Omit the directional key light (leave dome-only ambient). "
    "Use when the real scene has flat/ambient-only lighting.",
)
parser.add_argument(
    "--no-cube",
    action="store_true",
    dest="no_cube",
    help="Do not spawn the calibration cube (arm-only scene).",
)
parser.add_argument(
    "--dr-config",
    default=None,
    dest="dr_config",
    metavar="PATH",
    help="YAML file of DR augmentation steps to apply to the sim render in the "
    "'sim post+DR' panel.  Edit and save the file to hot-reload.  "
    "Example: so101_real/configs/dr_tuning.yaml",
)
parser.add_argument(
    "--ground-color",
    nargs=3,
    type=float,
    default=[0.9, 0.9, 0.9],
    dest="ground_color",
    metavar=("R", "G", "B"),
    help="Table surface diffuse RGB colour (default: 0.9 0.9 0.9 — white table). "
    "Values in [0, 1].",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# X11 / display resolution (mirrors tune_camera_pose.py)
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
        gui_procs = re.compile(r"gnome-shell|plasmashell|xfce4-session|Xorg|Xwayland")
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
                if not gui_procs.search(comm + " " + cmdline):
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

    print(f"[align] DISPLAY={os.environ.get('DISPLAY', '(not set)')}")
    print(f"[align] XAUTHORITY={os.environ.get('XAUTHORITY', '(not set)')}")


_resolve_x11(args_cli.display_sock)

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports after AppLauncher
# ---------------------------------------------------------------------------
import json
import subprocess
import sys
import termios
import tty
from pathlib import Path
from datetime import datetime as _dt

import cv2
import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from pxr import Gf, UsdGeom

_project_root = Path(__file__).resolve().parents[2]
if "ISAAC_LAB_WORKSPACE_PATH" not in os.environ:
    os.environ["ISAAC_LAB_WORKSPACE_PATH"] = str(_project_root)


def _resolve_bundle_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
    else:
        pin = _project_root / "scripts" / "pins" / "latest_bundle"
        if not pin.exists():
            raise FileNotFoundError(
                "No --bundle given and scripts/pins/latest_bundle does not exist. "
                "Pass --bundle PATH or run an export first."
            )
        p = pin.resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {p}")
    return p


from so101_rl.configurations.so101 import SO101_CFG
from so101_rl.configurations.camera import build_gripper_tiled_camera_cfg
from so101_rl.helpers.opencv_to_isaac_camera import apply_opencv_pinhole_distortion, load_intrinsics
from so101_rl.configurations.black_cube import BLACK_CUBE_CFG, CUBE_WIDTH

from so101_real.bundle import load_bundle
from so101_real.image_pipeline import build_deploy_pipeline

# Resolve and load the deploy bundle at module level so AlignSceneCfg can use
# the calibrated intrinsics when Isaac Sim builds the scene graph.
_bundle_dir = _resolve_bundle_dir(args_cli.bundle)
_bundle = load_bundle(_bundle_dir)
_bundle_intrinsics_path = _bundle.camera_intrinsics_path
if _bundle_intrinsics_path is None:
    raise ValueError(
        f"Bundle at {_bundle_dir} has no camera_intrinsics.yaml.\n"
        "Re-export from a training run that uses model: opencv_pinhole.\n"
        "Run: isaaclab.sh -p so101_rl/scripts/skrl/export_bundle.py ..."
    )
_bundle_intrinsics = load_intrinsics(_bundle_intrinsics_path)
from so101.utils.units import UNITS, JointUnitConverter, from_robot_config

# Local optimizer module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _camera_align.optimizer import (  # noqa: E402
    InsufficientFeaturesError,
    OrbMatcher,
    optimize as run_optimizer,
)
from _camera_align.dr_pipeline import (
    DRConfigWatcher,
    build_dr_aug_pipeline,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ImageNet stats — kept for reference only; normalization is now handled
# inside ResNet18SpatialSoftmaxFeatureExtractor, not in the image pipeline.
# No longer used for display inversion.


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class AlignSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
    camera: TiledCameraCfg = build_gripper_tiled_camera_cfg(  # type: ignore
        _bundle_intrinsics, args_cli.render_height, args_cli.render_width
    ).replace(data_types=["rgb"])
    cube: RigidObjectCfg = BLACK_CUBE_CFG  # type: ignore


# ---------------------------------------------------------------------------
# USD transform helpers (read + write)
# ---------------------------------------------------------------------------


def _xform_ops(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None, None
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op_type == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op
    return prim, translate_op, orient_op


def _read_xframe_transform(stage, prim_path: str):
    _, translate_op, orient_op = _xform_ops(stage, prim_path)
    if translate_op is None and orient_op is None:
        return None
    translate = (0.0, 0.0, 0.0)
    orient = (1.0, 0.0, 0.0, 0.0)
    if translate_op is not None:
        v = translate_op.Get()
        if v is not None:
            translate = (float(v[0]), float(v[1]), float(v[2]))
    if orient_op is not None:
        q = orient_op.Get()
        if q is not None:
            orient = (
                float(q.GetReal()),
                float(q.GetImaginary()[0]),
                float(q.GetImaginary()[1]),
                float(q.GetImaginary()[2]),
            )
    return translate, orient


def _write_xframe_transform(
    stage,
    prim_path: str,
    translate: tuple[float, float, float],
    orient_wxyz: tuple[float, float, float, float],
) -> bool:
    _, translate_op, orient_op = _xform_ops(stage, prim_path)
    if translate_op is None or orient_op is None:
        print(f"[align] WARNING: cannot write transform — missing xform ops on {prim_path}")
        return False
    translate_op.Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    current = orient_op.Get()
    w, x, y, z = (float(v) for v in orient_wxyz)
    if isinstance(current, Gf.Quatf):
        new_q = Gf.Quatf(w, Gf.Vec3f(x, y, z))
    else:
        new_q = Gf.Quatd(w, Gf.Vec3d(x, y, z))
    orient_op.Set(new_q)
    return True


def _print_transform(translate, orient_wxyz, out_yaml_path: str | None = None) -> None:
    tx, ty, tz = translate
    w, x, y, z = orient_wxyz
    print("\n" + "=" * 60)
    print("CameraXframe transform")
    print("=" * 60)
    print("\n--- Python literals (paste into camera.py) ---")
    print(f"CAMERA_TRANSLATE_VEC = ({tx}, {ty}, {tz})")
    print(f"CAMERA_ROTATION_QUAT_WXYZ = ({w}, {x}, {y}, {z})")
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
        print(f"[align] Written to: {p}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Image pipeline application + inverse for visualization
# ---------------------------------------------------------------------------


def _apply_deploy_pipeline_uint8(
    pipeline,
    sim_rgb_uint8: np.ndarray,
    device: torch.device,
    dr_steps: list | None = None,
) -> np.ndarray:
    """Apply the deploy pipeline to a (H, W, 3) uint8 RGB array for display.

    Returns the camera-appearance image as (H, W, 3) uint8 BGR at the **render
    resolution** (e.g. 640×360).  Resize steps are intentionally skipped —
    the policy CNN's input resolution is irrelevant for visual comparison; we
    want to see the full-resolution camera appearance, not a thumbnail.  DR
    augmentations (if provided) are inserted after Uint8ToFloatCHW.

    Parameters
    ----------
    pipeline:
        The deploy ImagePipeline loaded from the bundle.
    sim_rgb_uint8:
        Raw render from Isaac Sim, (H, W, 3) uint8 RGB.
    device:
        Torch device to run the pipeline on.
    dr_steps:
        Optional list of ImagePipelineStep augmentations to insert after the
        Uint8ToFloatCHW step and before the remaining pipeline steps (Clamp).
        Pass None or [] to skip augmentations.
    """
    from so101.utils.image_processing import Uint8ToFloatCHWPipelineStep

    t = torch.from_numpy(sim_rgb_uint8).to(device).unsqueeze(0)  # (1, H, W, 3) uint8

    with torch.no_grad():
        if dr_steps:
            # Split at the Uint8ToFloatCHW boundary so DR augmentations run
            # after conversion but before any remaining pipeline steps.
            split = next(
                (i + 1 for i, s in enumerate(pipeline.steps)
                 if isinstance(s, Uint8ToFloatCHWPipelineStep)),
                1,  # default: insert after first step
            )
            for step in pipeline.steps[:split]:
                t = step.process(t)
            for step in dr_steps:
                t = step.process(t)
            for step in pipeline.steps[split:]:
                t = step.process(t)
        else:
            for step in pipeline.steps:
                t = step.process(t)

    # Output is (1, 3, H, W) float in [0, 1] at render resolution — convert directly to uint8 BGR.
    arr = t[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3) float RGB
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

_COMPARE_MODES = ["blend", "abs-diff", "checkerboard", "side-by-side"]


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _compare_panel(
    real_match: np.ndarray,
    sim_match: np.ndarray,
    mode: str,
    alpha: float,
    out_w: int,
    out_h: int,
    checker_size: int = 32,
) -> np.ndarray:
    """Build the comparison panel (real vs sim-post) at output size."""
    H, W = real_match.shape[:2]
    if mode == "blend":
        composite = cv2.addWeighted(real_match, 1.0 - alpha, sim_match, alpha, 0.0)
    elif mode == "abs-diff":
        diff = cv2.absdiff(real_match, sim_match)
        composite = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=3.0), cv2.COLORMAP_INFERNO)
    elif mode == "checkerboard":
        composite = real_match.copy()
        n = checker_size
        for row in range(0, H, 2 * n):
            for col in range(0, W, 2 * n):
                composite[row : row + n, col : col + n] = sim_match[row : row + n, col : col + n]
                r2 = min(row + 2 * n, H)
                c2 = min(col + 2 * n, W)
                composite[row + n : r2, col + n : c2] = sim_match[row + n : r2, col + n : c2]
    elif mode == "side-by-side":
        composite = np.concatenate(
            [_label(real_match, "real"), _label(sim_match, "sim-post")], axis=1
        )
    else:
        composite = real_match
    return _resize_to(composite, out_w, out_h)


def _compute_diff_metrics(real_match: np.ndarray, sim_match: np.ndarray) -> tuple[float, float]:
    """Return (mean_abs_diff [0..1], pearson_ncc [-1..1]) at the policy resolution."""
    a = real_match.astype(np.float32) / 255.0
    b = sim_match.astype(np.float32) / 255.0
    mad = float(np.mean(np.abs(a - b)))
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    ncc = float(np.dot(af, bf) / denom) if denom > 0 else 0.0
    return mad, ncc


def _build_full_frame(
    real_panel: np.ndarray,
    sim_raw_panel: np.ndarray,
    sim_post_panel: np.ndarray,
    compare_panel: np.ndarray,
    panel_w: int,
    panel_h: int,
    hud_lines: list[str],
    post_label: str = "sim post-pipeline",
) -> np.ndarray:
    """Compose the 3-up top row + comparison panel + HUD."""
    top_row = np.concatenate(
        [
            _label(real_panel, "real"),
            _label(sim_raw_panel, "sim raw"),
            _label(sim_post_panel, post_label),
        ],
        axis=1,
    )
    full = np.concatenate([top_row, compare_panel], axis=0)
    # Burn HUD lines into the bottom of the comparison panel.
    y = full.shape[0] - 8 - 14 * (len(hud_lines) - 1)
    for line in hud_lines:
        cv2.putText(full, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(full, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 14
    return full


# ---------------------------------------------------------------------------
# ffplay display + raw terminal (copied verbatim from tune_camera_pose.py)
# ---------------------------------------------------------------------------


class FfplayDisplay:
    def __init__(self, width: int, height: int, title: str, display_width: int | None = None):
        self.width = width
        self.height = height
        vf_args: list[str] = []
        if display_width and display_width < width:
            scaled_h = int(height * display_width / width)
            scaled_h += scaled_h % 2
            display_w = display_width + (display_width % 2)
            vf_args = ["-vf", f"scale={display_w}:{scaled_h}"]
        cmd = [
            "ffplay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", "30",
            "-i", "pipe:0",
            "-window_title", title,
            "-an",
            "-autoexit",
            *vf_args,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame_rgb: np.ndarray) -> bool:
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


class _RawTerminal:
    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # disables canonical mode + echo but preserves OPOST
        # Isaac Sim shutdown can call os._exit(), bypassing finally blocks.
        # Register an atexit handler as a best-effort fallback (runs for sys.exit()).
        import atexit
        atexit.register(self._restore_tty)
        return self

    def _restore_tty(self) -> None:
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, self._old)
        except Exception:
            pass

    def __exit__(self, *_):
        import atexit
        atexit.unregister(self._restore_tty)
        self._restore_tty()

    def read_key(self) -> str | None:
        import select

        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
# Robot bridge subprocess (copied from tune_camera_pose.py)
# ---------------------------------------------------------------------------


class RobotBridgeReader:
    def __init__(self, robot_config_path: str, lerobot_python: str | None = None) -> None:
        import threading

        if lerobot_python is None:
            candidates = [
                "/opt/miniforge3/bin/python3.12",
                "/opt/miniforge3/bin/python3",
                str(Path.home() / ".conda" / "envs" / "lerobot" / "bin" / "python"),
                "/opt/conda/bin/python3.12",
            ]
            lerobot_python = next((p for p in candidates if Path(p).exists()), candidates[0])

        bridge_script = Path(__file__).resolve().parent / "robot_bridge.py"
        if not bridge_script.exists():
            raise FileNotFoundError(f"robot_bridge.py not found at {bridge_script}")
        if not Path(lerobot_python).exists():
            raise FileNotFoundError(
                f"lerobot Python not found: {lerobot_python}\nPass --lerobot-python."
            )

        self._latest = {n: 0.0 for n in _JOINT_NAMES}
        self._lock = threading.Lock()
        self._running = True

        passthrough = {
            "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG",
            "LC_ALL", "LC_CTYPE", "DISPLAY", "XAUTHORITY", "LD_LIBRARY_PATH",
        }
        bridge_env = {k: v for k, v in os.environ.items() if k in passthrough}
        for var in (
            "PYTHONPATH", "PYTHONHOME", "ISAAC_LAB_WORKSPACE_PATH",
            "CARB_APP_PATH", "EXP_PATH", "OMNI_KIT_ALLOW_ROOT",
        ):
            bridge_env.pop(var, None)

        self._proc = subprocess.Popen(
            [lerobot_python, str(bridge_script), "--robot-config", robot_config_path],
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1, env=bridge_env,
        )

        msg = {}
        while True:
            line = self._proc.stdout.readline()
            if not line:
                self._proc.wait()
                raise RuntimeError(
                    f"robot_bridge.py exited before sending 'ready' "
                    f"(exit code {self._proc.returncode})."
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                break
            except json.JSONDecodeError:
                print(f"[bridge] {line}", file=sys.stderr, flush=True)
                continue
        if msg.get("status") != "ready":
            raise RuntimeError(f"Unexpected bridge first message: {msg!r}")
        print(f"[align] Robot bridge ready (pid {self._proc.pid})")

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
        with self._lock:
            return [self._latest.get(n, 0.0) for n in _JOINT_NAMES]

    def disconnect(self) -> None:
        self._running = False
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def _load_joint_pose_yaml(
    path: str | None,
    robot_config_path: str | None = None,
) -> dict[str, float]:
    """Load joint pose from YAML and return canonical-radian values.

    Reads an optional top-level ``unit:`` key (default ``"rad"``) and applies
    the appropriate :class:`JointUnitConverter` so callers always receive
    canonical radians regardless of the unit stored in the file.

    Raises ``ValueError`` at startup (before Isaac Sim launches) if the
    requested unit requires ``--robot-config`` data that was not supplied.
    """
    if path is None:
        return {n: 0.0 for n in _JOINT_NAMES}

    with open(path) as fh:
        data = yaml.safe_load(fh)

    unit: str = str(data.get("unit", "rad")).strip()
    if unit not in UNITS:
        raise ValueError(
            f"calibration_pose.yaml: unknown unit {unit!r}. "
            f"Valid values: {UNITS}."
        )

    # Units that need extra data from robot.yaml.
    needs_limits = unit == "norm"
    needs_calib = unit in ("lrad", "ldeg")
    if (needs_limits or needs_calib) and robot_config_path is None:
        raise ValueError(
            f"calibration_pose.yaml specifies unit={unit!r} which requires "
            "joint calibration/limits from robot.yaml. "
            "Pass --robot-config <path> or change unit to 'rad'/'deg'."
        )

    # Build a converter appropriate for the requested unit.
    if needs_limits or needs_calib:
        # Import here (inside Isaac Sim env) — so101_real.robot is heavy.
        from so101_real.robot import RobotConfig  # noqa: PLC0415

        cfg = RobotConfig.load(robot_config_path)
        joint_names_ordered = _JOINT_NAMES
        lower_rad = [cfg.joint_limits[n].lower_rad for n in joint_names_ordered] if needs_limits else None
        upper_rad = [cfg.joint_limits[n].upper_rad for n in joint_names_ordered] if needs_limits else None
        calib_lero_scale = [cfg.joint_limits[n].lero_scale for n in joint_names_ordered] if needs_calib else None
        calib_lero_offset = [cfg.joint_limits[n].lero_offset_rad for n in joint_names_ordered] if needs_calib else None
        converter = from_robot_config(
            joint_names_ordered,
            lower_rad=lower_rad,
            upper_rad=upper_rad,
            lero_scale=calib_lero_scale,
            lero_offset_rad=calib_lero_offset,
        )
    else:
        # rad / deg need no extra data.
        converter = JointUnitConverter(_JOINT_NAMES)

    result: dict[str, float] = {}
    for i, name in enumerate(_JOINT_NAMES):
        raw = float(data.get(name, 0.0))
        result[name] = converter.to_canonical_rad(raw, unit, joint_index=i)
    return result


def _load_real_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(f"[align] ERROR: could not read real image: {path}")
    return img


def _grip_zone_world_pose(
    robot: "Articulation",
    ee_idx: int,
    grip_zone_offset_ee: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos, quat_wxyz) of the grip-zone centre in world frame, shape (1,3)/(1,4)."""
    gripper_pos_w = robot.data.body_pos_w[:, ee_idx : ee_idx + 1, :].squeeze(1)  # (1,3)
    gripper_quat_w = robot.data.body_quat_w[:, ee_idx : ee_idx + 1, :].squeeze(1)  # (1,4)
    gz_pos = gripper_pos_w + quat_apply(gripper_quat_w, grip_zone_offset_ee.unsqueeze(0))
    return gz_pos, gripper_quat_w


def _compute_grip_zone_offset(stage, device: torch.device) -> torch.Tensor | None:
    """Compute the grip-zone offset vector in the gripper EE local frame.

    Mirrors ``GripZoneOffsetEnvMetricStep._cache_gripperframe_transform()``
    from ``env_metric_pipeline.py``, with ``height_scale = 1.0`` (no domain
    randomisation) and the same ``_GZ_CLEARANCE = 0.001 m`` constant.

    The result is a ``(3,)`` float32 tensor expressed in the gripper body's
    local frame, suitable for ``quat_apply(gripper_quat_w, offset)`` to get
    the world-frame grip-zone centre.

    Returns ``None`` if the ``gripperframe`` prim has not yet been spawned.
    """
    gripperframe_path = "/World/envs/env_0/Robot/gripper/gripperframe"
    prim = stage.GetPrimAtPath(gripperframe_path)
    if not prim.IsValid():
        print(f"[align] gripperframe prim not found at {gripperframe_path}")
        return None

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    local_xform = UsdGeom.Xformable(prim).GetLocalTransformation()
    t = local_xform.ExtractTranslation()
    q = local_xform.ExtractRotationQuat()
    qi = q.GetImaginary()

    pos = torch.tensor(
        [t[0] * meters_per_unit, t[1] * meters_per_unit, t[2] * meters_per_unit],
        device=device,
        dtype=torch.float32,
    )
    # Isaac Lab uses wxyz; USD Quatd stores (real, imaginary) so GetReal()==w.
    quat = torch.tensor(
        [float(q.GetReal()), float(qi[0]), float(qi[1]), float(qi[2])],
        device=device,
        dtype=torch.float32,
    )
    quat = quat / quat.norm()

    tooth_normal = quat_apply(
        quat.unsqueeze(0),
        torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32),
    ).squeeze(0)

    # offset_mag = cube_half_height + 1 mm clearance (height_scale = 1.0, no DR)
    offset_mag = (CUBE_WIDTH / 2.0) + 0.001
    return pos + tooth_normal * offset_mag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not args_cli.no_robot and args_cli.robot_config is None:
        print("ERROR: --robot-config is required unless --no-robot is set.")
        sys.exit(1)

    # ── Deploy bundle (resolved at module level for AlignSceneCfg) ─────────
    bundle_dir = _bundle_dir
    bundle = _bundle
    print(f"[align] Loading deploy bundle: {bundle_dir}")
    image_pipeline = build_deploy_pipeline(bundle)
    policy_h, policy_w = bundle.image_height, bundle.image_width
    print(
        f"[align] Deploy pipeline: {len(image_pipeline.steps)} steps, "
        f"output={policy_w}x{policy_h}"
    )

    pipeline_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── DR augmentation config ───────────────────────────────────────────────
    dr_steps: list = []
    dr_watcher: DRConfigWatcher | None = None
    if args_cli.dr_config is not None:
        dr_steps = build_dr_aug_pipeline(args_cli.dr_config, pipeline_device)
        dr_watcher = DRConfigWatcher(args_cli.dr_config)
        print(
            f"[align] DR config: {args_cli.dr_config} "
            f"({len(dr_steps)} active steps)"
        )

    # ── Sim setup ───────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.3, 0.0, 0.3])

    # Table — same geometry as TABLE_CFG in the training env (center (0.45, 0, -0.5),
    # size (2, 2, 1) m), so the top surface is at z = 0 and the wrist-camera view
    # matches the training distribution.  --ground-color overrides the diffuse colour
    # if the real workspace differs from the default white.
    table_color = tuple(float(c) for c in args_cli.ground_color)
    table_shape_cfg = sim_utils.CuboidCfg(
        size=(2.0, 2.0, 1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=table_color,
            roughness=0.8,
            metallic=0.0,
        ),
    )
    table_shape_cfg.func("/World/Table", table_shape_cfg, translation=(0.45, 0.0, -0.5))

    # Lights — mirrors the training env so the sim-post-pipeline feed is
    # in the same distribution the policy was trained on.
    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    dome_light_cfg.func("/World/Light", dome_light_cfg)
    if not args_cli.no_key_light:
        key_light_cfg = sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(1.0, 0.95, 0.85),
            angle=0.53,
        )
        key_light_cfg.func(
            "/World/KeyLight",
            key_light_cfg,
            orientation=(0.8924, 0.2392, -0.3604, 0.0966),
        )

    scene_cfg = AlignSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[align] Scene ready.")

    robot: Articulation = scene["robot"]
    camera: TiledCamera = scene["camera"]

    # Apply lens distortion (OmniLensDistortionOpenCvPinholeAPI) to each spawned camera
    # prim using intrinsics from the bundle, which were already validated at startup.
    import omni.usd as _omni_usd
    _stage = _omni_usd.get_context().get_stage()
    for _prim_path in camera._view.prim_paths:
        apply_opencv_pinhole_distortion(_stage.GetPrimAtPath(_prim_path), _bundle_intrinsics)
    del _omni_usd, _stage, _prim_path

    cube: RigidObject = scene["cube"]
    joint_indices, _ = robot.find_joints(_JOINT_NAMES)
    num_joints = len(joint_indices)
    ee_idx_list, _ = robot.find_bodies(["gripper"])
    ee_idx: int = ee_idx_list[0]
    sim_device = sim.device

    # ── Real-robot connection or fixed pose ─────────────────────────────────
    real_robot: RobotBridgeReader | None = None
    fixed_joint_pos: torch.Tensor | None = None
    if args_cli.no_robot:
        pose_dict = _load_joint_pose_yaml(args_cli.joint_pose, robot_config_path=args_cli.robot_config)
        fixed_joint_pos = torch.zeros((1, num_joints), dtype=torch.float32, device=sim_device)
        for i, name in enumerate(_JOINT_NAMES):
            fixed_joint_pos[0, i] = pose_dict[name]
        print(f"[align] --no-robot mode. Fixed joint pose: {pose_dict}")
    else:
        real_robot = RobotBridgeReader(args_cli.robot_config, lerobot_python=args_cli.lerobot_python)

    # ── Real reference image ────────────────────────────────────────────────
    real_bgr = _load_real_image(args_cli.real_image)
    if real_bgr is None:
        print(f"[align] WARNING: real image missing — using grey placeholder.")
        real_bgr = np.full(
            (args_cli.render_height, args_cli.render_width, 3), 128, dtype=np.uint8
        )

    # ── USD stage + camera prim ─────────────────────────────────────────────
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    camera_xframe_path = (
        "/World/envs/env_0/Robot/gripper/mountscrew/camera_mount/CameraXframe"
    )

    # ── Grip-zone offset + cube init ─────────────────────────────────────────
    # Compute the grip-zone offset once from the USD gripperframe prim, then
    # re-glue the cube to that world-frame position every tick.
    grip_zone_offset_ee: torch.Tensor | None = None
    if not args_cli.no_cube:
        grip_zone_offset_ee = _compute_grip_zone_offset(stage, sim_device)
        if grip_zone_offset_ee is None:
            print("[align] WARNING: gripperframe prim not found; disabling cube target.")
            args_cli.no_cube = True
    if args_cli.no_cube:
        # Teleport the cube off-screen so it never appears in the camera feeds.
        _offscreen = torch.tensor(
            [[0.0, 0.0, -100.0, 1.0, 0.0, 0.0, 0.0]], device=sim_device
        )
        cube.write_root_pose_to_sim(_offscreen)
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim_device))
        cube.write_data_to_sim()

    # ── Display geometry ────────────────────────────────────────────────────
    panel_w = args_cli.render_width
    panel_h = args_cli.render_height
    full_w = panel_w * 3
    full_h = panel_h * 2

    ffplay = FfplayDisplay(
        width=full_w,
        height=full_h,
        title="align-camera (real | sim-raw | sim-post + comparison)",
        display_width=args_cli.display_width,
    )

    # ── Loop state ──────────────────────────────────────────────────────────
    alpha = 0.5
    compare_idx = 0
    optim_status = "idle"
    dr_active: bool = args_cli.dr_config is not None
    dr_last_reload: str = ""

    print(
        "[align] Hotkeys: [ ] alpha   c compare-mode   o optimize   "
        "a DR-toggle   d dump-metrics   s snapshot   r reload-real   q quit"
    )

    with _RawTerminal() as kbd:
        try:
            while simulation_app.is_running():
                sim.step()
                scene.update(sim.get_physics_dt())

                # ── Drive joints ────────────────────────────────────────────
                if real_robot is not None:
                    raw = real_robot.read_joints()
                    q = torch.zeros((1, num_joints), dtype=torch.float32, device=sim_device)
                    for i in range(num_joints):
                        q[0, i] = float(raw[i])
                else:
                    q = fixed_joint_pos
                robot.write_joint_position_to_sim(q, joint_ids=joint_indices)
                robot.write_joint_velocity_to_sim(torch.zeros_like(q), joint_ids=joint_indices)
                robot.write_data_to_sim()

                # ── Pin cube to grip zone ────────────────────────────────────────
                if not args_cli.no_cube and grip_zone_offset_ee is not None:
                    _gz_pos, _gz_quat = _grip_zone_world_pose(
                        robot, ee_idx, grip_zone_offset_ee, sim_device
                    )
                    cube.write_root_pose_to_sim(torch.cat([_gz_pos, _gz_quat], dim=-1))
                    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim_device))
                    cube.write_data_to_sim()

                # ── Sim camera frame ────────────────────────────────────────
                rgb_tensor = camera.data.output.get("rgb")
                if rgb_tensor is None or rgb_tensor.shape[0] == 0:
                    continue
                sim_rgb = rgb_tensor[0].cpu().numpy()  # (H, W, 3) uint8 RGB
                sim_raw_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

                # ── Hot-reload DR config ─────────────────────────────────────
                if dr_watcher is not None and dr_watcher.changed:
                    try:
                        dr_steps = build_dr_aug_pipeline(
                            args_cli.dr_config, pipeline_device
                        )
                        dr_last_reload = _dt.now().strftime("%H:%M:%S")
                        print(
                            f"\r[align] DR config reloaded: "
                            f"{len(dr_steps)} active steps  ",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"\r[align] DR reload error: {exc}  ", flush=True)
                    dr_watcher.changed = False

                # ── Sim post-pipeline (policy view) ─────────────────────────
                active_dr = dr_steps if (dr_active and dr_steps) else None
                sim_post_bgr_native = _apply_deploy_pipeline_uint8(
                    image_pipeline, sim_rgb, pipeline_device, dr_steps=active_dr
                )  # (render_h, render_w, 3) BGR uint8 — Resize step is skipped for display

                # Real frame at render resolution for diff metrics (matches sim post output).
                real_match_native = cv2.resize(
                    real_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA
                )

                # ── Diff metrics ────────────────────────────────────────────
                mad, ncc = _compute_diff_metrics(real_match_native, sim_post_bgr_native)

                # ── Display panels (uniform size) ───────────────────────────
                real_panel = _resize_to(real_bgr, panel_w, panel_h)
                sim_raw_panel = _resize_to(sim_raw_bgr, panel_w, panel_h)
                post_label = "sim post+DR" if (dr_active and dr_steps) else "sim post-pipeline"
                sim_post_panel = _resize_to(sim_post_bgr_native, panel_w, panel_h)

                compare_mode = _COMPARE_MODES[compare_idx]
                compare_panel = _compare_panel(
                    real_match_native, sim_post_bgr_native,
                    compare_mode, alpha, full_w, panel_h,
                )

                if args_cli.dr_config is None:
                    dr_hud = "DR: not configured"
                elif dr_active:
                    reload_tag = f"  reloaded={dr_last_reload}" if dr_last_reload else ""
                    dr_hud = f"DR: ON ({len(dr_steps)} steps{reload_tag})"
                else:
                    dr_hud = "DR: OFF"
                hud_lines = [
                    f"compare={compare_mode}  alpha={alpha:.2f}  "
                    f"MAD={mad:.4f}  NCC={ncc:+.3f}  optim={optim_status}",
                    f"{dr_hud}   [ ] alpha   c mode   a DR-toggle   o optimize   d dump   s snap   r reload   q quit",
                ]
                full = _build_full_frame(
                    real_panel, sim_raw_panel, sim_post_panel, compare_panel,
                    panel_w, panel_h, hud_lines,
                    post_label=post_label,
                )
                full_rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
                if not ffplay.write(full_rgb):
                    print("[align] ffplay window closed — exiting.")
                    break

                # ── Keys ───────────────────────────────────────────────────
                ch = kbd.read_key()
                if ch == "q":
                    break
                elif ch == "[":
                    alpha = max(0.0, alpha - 0.05)
                elif ch == "]":
                    alpha = min(1.0, alpha + 0.05)
                elif ch == "c":
                    compare_idx = (compare_idx + 1) % len(_COMPARE_MODES)
                elif ch == "a":
                    if args_cli.dr_config is None:
                        print("\r[align] DR not configured (pass --dr-config PATH)  ", flush=True)
                    else:
                        dr_active = not dr_active
                        print(
                            f"\r[align] DR augmentation: {'ON' if dr_active else 'OFF'}  ",
                            flush=True,
                        )
                elif ch == "r":
                    new_img = _load_real_image(args_cli.real_image)
                    if new_img is not None:
                        real_bgr = new_img
                        print(f"\r[align] Reloaded: {args_cli.real_image}", flush=True)
                elif ch == "d":
                    print()
                    print(f"[align] MAD={mad:.4f}  NCC={ncc:+.3f}")
                    result = _read_xframe_transform(stage, camera_xframe_path)
                    if result is not None:
                        _print_transform(*result)
                elif ch == "s":
                    result = _read_xframe_transform(stage, camera_xframe_path)
                    if result is not None:
                        print()
                        _print_transform(*result, out_yaml_path=args_cli.out_yaml)
                elif ch == "o":
                    optim_status = "running"
                    print("\n[align] Running ORB + Nelder-Mead optimizer...")
                    # ORB runs at render resolution (640×360), not policy
                    # resolution (192×108) — low-res images yield zero keypoints.
                    real_orb_bgr = _resize_to(
                        real_bgr, args_cli.render_width, args_cli.render_height
                    )
                    _run_optimizer_inline(
                        stage=stage,
                        camera_xframe_path=camera_xframe_path,
                        sim=sim,
                        scene=scene,
                        camera=camera,
                        robot=robot,
                        joint_indices=joint_indices,
                        joint_target=q,
                        real_orb_bgr=real_orb_bgr,
                        translation_bound_m=args_cli.optim_translation_bound_m,
                        rotation_bound_deg=args_cli.optim_rotation_bound_deg,
                        max_evals=args_cli.optim_max_evals,
                        min_matches=args_cli.optim_min_matches,
                        ffplay=ffplay,
                        real_panel=_resize_to(real_bgr, panel_w, panel_h),
                        panel_w=panel_w,
                        panel_h=panel_h,
                        cube=cube if not args_cli.no_cube else None,
                        ee_idx=ee_idx if not args_cli.no_cube else None,
                        grip_zone_offset_ee=grip_zone_offset_ee,
                    )
                    optim_status = "done"
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[align] Exit — final CameraXframe transform:")
            result = _read_xframe_transform(stage, camera_xframe_path)
            if result is not None:
                _print_transform(*result, out_yaml_path=args_cli.out_yaml)
            ffplay.close()
            if real_robot is not None:
                real_robot.disconnect()
                print("[align] Real robot disconnected.")


# ---------------------------------------------------------------------------
# Optimizer driver
# ---------------------------------------------------------------------------


def _push_optim_frame(
    sim_bgr: np.ndarray,
    real_panel: np.ndarray,
    panel_w: int,
    panel_h: int,
    ffplay: FfplayDisplay,
    label: str,
) -> None:
    """Write a 3-panel optimizer frame (real | sim-eval | side-by-side) to ffplay."""
    sim_panel = _resize_to(sim_bgr, panel_w, panel_h)
    real_resized = _resize_to(real_panel, panel_w, panel_h)

    # Side-by-side diff of real vs current sim eval candidate.
    sbs = np.concatenate([_label(real_resized, "real"), _label(sim_panel, "sim eval")], axis=1)
    sbs = _resize_to(sbs, panel_w * 3, panel_h)

    top_row = np.concatenate(
        [_label(real_resized, "real"), sim_panel, _label(sim_panel, "sim eval")],
        axis=1,
    )
    full = np.concatenate([top_row, sbs], axis=0)

    # Overlay the status label.
    cv2.putText(full, label, (8, full.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(full, label, (8, full.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 80), 1, cv2.LINE_AA)

    ffplay.write(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))


def _run_optimizer_inline(
    *,
    stage,
    camera_xframe_path: str,
    sim,
    scene,
    camera: TiledCamera,
    robot: Articulation,
    joint_indices,
    joint_target: torch.Tensor,
    real_orb_bgr: np.ndarray,
    translation_bound_m: float,
    rotation_bound_deg: float,
    max_evals: int,
    min_matches: int,
    ffplay: FfplayDisplay,
    real_panel: np.ndarray,
    panel_w: int,
    panel_h: int,
    cube: RigidObject | None = None,
    ee_idx: int | None = None,
    grip_zone_offset_ee: torch.Tensor | None = None,
) -> None:
    """Step the sim with candidate transforms and minimize ORB feature error.

    ORB runs on the **raw** sim render (render_width x render_height, e.g.
    640x360) and the correspondingly-resized real image.  The post-pipeline
    192x108 thumbnail is far too small for reliable keypoint detection.
    """
    base = _read_xframe_transform(stage, camera_xframe_path)
    if base is None:
        print("[align] Cannot optimize — CameraXframe prim unavailable.")
        return
    base_translate, base_orient = base
    print(
        f"[align]   start  translate={base_translate}  orient_wxyz={base_orient}"
    )

    _eval_state = {"n": 0}

    def _render(translate, orient_wxyz) -> np.ndarray:
        if not _write_xframe_transform(stage, camera_xframe_path, translate, orient_wxyz):
            raise RuntimeError("Failed to write candidate transform")
        # 3 sim steps at dt=1/60s = 0.05s > camera update_period (1/30s = 0.033s),
        # guaranteeing at least one full camera refresh cycle.
        for _ in range(3):
            robot.write_joint_position_to_sim(joint_target, joint_ids=joint_indices)
            robot.write_joint_velocity_to_sim(
                torch.zeros_like(joint_target), joint_ids=joint_indices
            )
            robot.write_data_to_sim()
            if cube is not None and ee_idx is not None and grip_zone_offset_ee is not None:
                _gz_pos, _gz_quat = _grip_zone_world_pose(
                    robot, ee_idx, grip_zone_offset_ee, joint_target.device
                )
                cube.write_root_pose_to_sim(torch.cat([_gz_pos, _gz_quat], dim=-1))
                cube.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=joint_target.device)
                )
                cube.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        rgb_tensor = camera.data.output.get("rgb")
        if rgb_tensor is None or rgb_tensor.shape[0] == 0:
            raise RuntimeError("Camera produced no frame during optimization")
        sim_rgb = rgb_tensor[0].cpu().numpy()
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        # Push this candidate frame to ffplay so the display updates each eval.
        _eval_state["n"] += 1
        _push_optim_frame(
            sim_bgr, real_panel, panel_w, panel_h, ffplay,
            label=f"OPTIMIZING  eval {_eval_state['n']} / {max_evals}",
        )
        return sim_bgr

    orb_h, orb_w = real_orb_bgr.shape[:2]
    print(f"[align]   ORB image size: {orb_w}x{orb_h}")
    matcher = OrbMatcher(real_orb_bgr)
    n_real_kp = len(matcher.real_kp)
    print(f"[align]   ORB keypoints in real image: {n_real_kp}")

    def _progress(idx: int, loss: float, match) -> None:
        if idx == 1 or idx % 5 == 0:
            print(
                f"[align]   eval {idx:>3}  loss={loss:7.3f}  "
                f"reproj_err={match.mean_reprojection_error_px:6.2f}px  "
                f"matches={match.n_matches}  inliers={match.n_inliers}"
            )

    try:
        result = run_optimizer(
            matcher,
            _render,
            base_translate,
            base_orient,
            translation_bound_m=translation_bound_m,
            rotation_bound_deg=rotation_bound_deg,
            max_evals=max_evals,
            min_matches=min_matches,
            progress_cb=_progress,
        )
    except InsufficientFeaturesError as exc:
        print(f"[align] Optimizer aborted: {exc}")
        # Restore base transform and display the restored frame.
        _write_xframe_transform(stage, camera_xframe_path, base_translate, base_orient)
        try:
            restored_bgr = _render(base_translate, base_orient)
            _push_optim_frame(
                restored_bgr, real_panel, panel_w, panel_h, ffplay,
                label="OPTIMIZER ABORTED — restored",
            )
        except Exception:
            pass
        return

    print(
        f"[align] Done. evals={result.n_evals}  final_loss={result.final_loss:.3f}  "
        f"final_reproj_err={result.final_match.mean_reprojection_error_px:.2f}px  "
        f"inliers={result.final_match.n_inliers}/{result.final_match.n_matches}"
    )
    print(f"[align]   delta(t, r_deg) = {np.array2string(result.best_delta, precision=4)}")

    # Write the best transform and re-render so the display shows the best
    # result, not whichever candidate NM happened to evaluate last.
    _write_xframe_transform(
        stage, camera_xframe_path, result.best_translate, result.best_orient_wxyz
    )
    try:
        best_bgr = _render(result.best_translate, result.best_orient_wxyz)
        _push_optim_frame(
            best_bgr, real_panel, panel_w, panel_h, ffplay,
            label=f"DONE  loss={result.final_loss:.3f}  inliers={result.final_match.n_inliers}",
        )
    except Exception:
        pass
    _print_transform(result.best_translate, result.best_orient_wxyz)


if __name__ == "__main__":
    main()
    simulation_app.close()

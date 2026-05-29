"""configure-camera — apply V4L2 controls from a YAML config file."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def _load_config(path: str) -> tuple[str, dict[str, int]]:
    """Load camera_v4l2.yaml and return (device, controls)."""
    data = yaml.safe_load(Path(path).read_text())
    if "device" not in data:
        raise ValueError(f"'device' key missing from {path}")
    if "controls" not in data or not isinstance(data["controls"], dict):
        raise ValueError(f"'controls' mapping missing from {path}")
    return data["device"], data["controls"]


def cmd_configure_camera(args) -> None:
    device, controls = _load_config(args.camera_config)

    if not controls:
        raise ValueError(f"No controls defined in {args.camera_config}")

    print(f"[configure-camera] Applying {len(controls)} control(s) to {device}")

    for name, value in controls.items():
        ctrl_arg = f"{name}={value}"
        cmd = ["v4l2-ctl", "-d", device, f"--set-ctrl={ctrl_arg}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"v4l2-ctl failed for '{ctrl_arg}': {result.stderr.strip()}"
            )
        print(f"  {ctrl_arg}")

    if not args.no_verify:
        control_names = ",".join(controls.keys())
        cmd = ["v4l2-ctl", "-d", device, f"--get-ctrl={control_names}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"v4l2-ctl --get-ctrl failed: {result.stderr.strip()}")
        print(f"\n[configure-camera] Verified settings on {device}:")
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")


def add_parser(sub) -> None:
    p = sub.add_parser(
        "configure-camera",
        help="Apply V4L2 controls from a YAML config (e.g. exposure, anti-flicker)",
    )
    p.add_argument(
        "--camera-config",
        required=True,
        dest="camera_config",
        metavar="PATH",
        help="Path to camera_v4l2.yaml",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        dest="no_verify",
        help="Skip reading back the applied settings",
    )
    p.set_defaults(func=cmd_configure_camera)

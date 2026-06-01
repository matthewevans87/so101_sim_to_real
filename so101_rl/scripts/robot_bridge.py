"""robot_bridge.py — Joint position bridge for the lerobot conda env.

Run this script with the lerobot Python interpreter.  It connects to the
physical SO-101, reads joint positions at ~30 Hz, and prints them as newline-
delimited JSON to stdout.  tune_camera_pose.py spawns this as a subprocess and
reads from the pipe so that the Isaac Lab process (env_isaaclab) does not need
lerobot installed.

Usage (normally invoked automatically by tune_camera_pose.py):
    /path/to/lerobot/python so101_rl/scripts/robot_bridge.py \
        --robot-config so101_real/configs/robot.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make sure the project root is on the path so so101_real is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from so101_real.robot import RobotConfig, So101Robot

_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

_HZ = 30

parser = argparse.ArgumentParser(
    description="SO-101 robot joint bridge (stdout JSON stream)"
)
parser.add_argument(
    "--robot-config",
    required=True,
    metavar="PATH",
    help="Path to so101_real robot config YAML",
)
args = parser.parse_args()

config = RobotConfig.load(args.robot_config)
robot = So101Robot(config=config, joint_names=_JOINT_NAMES)
robot.connect()

# Signal readiness to parent process
print(json.dumps({"status": "ready"}), flush=True)

try:
    while True:
        q = robot.read_joints()
        data = {name: float(q[i]) for i, name in enumerate(_JOINT_NAMES)}
        print(json.dumps(data), flush=True)
        time.sleep(1.0 / _HZ)
except KeyboardInterrupt:
    pass
except BrokenPipeError:
    pass  # parent process has exited
finally:
    robot.disconnect()

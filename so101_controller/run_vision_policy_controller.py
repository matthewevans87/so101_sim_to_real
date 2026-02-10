#!/usr/bin/env python3
"""
Example script demonstrating how to use StormVisionPolicy with PolicyController.

This script shows how to:
1. Connect to a USB webcam (cross-platform: Linux/macOS)
2. Load a pretrained StormVisionPolicy checkpoint
3. Run the vision-based policy on the robot
"""

import math
import sys
from pathlib import Path

from sympy import beta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from so101_controller.StaticPositionPolicy import (
    StaticPositionPolicy,
    get_home_position_policy,
    get_safe_position_policy,
)
from so101_controller.PolicyConfiguration import PolicyConfiguration
import torch
import argparse
from so101_controller.StormVisionPolicy import VisionFcPolicy, VisionConvPolicy
from so101_controller.PolicyController import PolicyController
from so101_controller.ControllerConfiguration import ControllerConfiguration
from so101_controller.So101RobotInterface import So101RobotInterface
from so101_controller.CameraSource import CameraSource, list_available_cameras


def get_vision_fc_policy(
    policy_config: PolicyConfiguration, device: torch.device
) -> VisionFcPolicy:

    # 1) Load exported checkpoint (container dict)
    ckpt = torch.load(policy_config.policy_path, map_location=device)

    # handle both "container dict" and "raw state_dict" cases
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        obs_dim = ckpt.get("obs_dim", policy_config.obs_dim)
        act_dim = ckpt.get("act_dim", policy_config.act_dim)
        hidden1 = ckpt.get("hidden1", 256)
        hidden2 = ckpt.get("hidden2", 128)
        hidden3 = ckpt.get("hidden3", 64)
    else:
        state_dict = ckpt
        obs_dim = policy_config.obs_dim
        act_dim = policy_config.act_dim
        hidden1 = 256
        hidden2 = 128
        hidden3 = 64

    # 2) Build policy with matching dims
    policy = VisionFcPolicy(
        num_joints=len(policy_config.dof_names),
        act_dim=act_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        device=device,
    ).to(device)

    # Load only the MLP head weights (fc1, fc2, fc3, fc4)
    # The ResNet18 backbone is already loaded from torchvision in __init__
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    return policy


def get_vision_conv_policy(
    policy_config: PolicyConfiguration, device: torch.device
) -> VisionConvPolicy:

    # 1) Load exported checkpoint (container dict)
    ckpt = torch.load(policy_config.policy_path, map_location=device)

    # handle both "container dict" and "raw state_dict" cases
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        obs_dim = ckpt.get("obs_dim", policy_config.obs_dim)
        act_dim = ckpt.get("act_dim", policy_config.act_dim)
        hidden1 = ckpt.get("hidden1", 256)
        hidden2 = ckpt.get("hidden2", 128)
        hidden3 = ckpt.get("hidden3", 64)
    else:
        state_dict = ckpt
        obs_dim = policy_config.obs_dim
        act_dim = policy_config.act_dim
        hidden1 = 256
        hidden2 = 128
        hidden3 = 64

    # 2) Build policy with matching dims
    policy = VisionConvPolicy(
        num_joints=len(policy_config.dof_names),
        act_dim=act_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        device=device,
    ).to(device)

    # Load only the MLP head weights (fc1, fc2, fc3, fc4)
    # The ResNet18 backbone is already loaded from torchvision in __init__
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Run vision-based policy on SO-101 robot with USB webcam"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (e.g., 0, 1, 2). If not specified, will list available cameras.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint file",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["fc", "conv"],
        default="conv",
        help="Vision policy type: 'fc' for 512-D features (avgpool), 'conv' for 1024-D features (spatial softmax)",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for robot connection",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="path/to/so101_new_calib.urdf",
        help="Path to robot URDF file",
    )
    args = parser.parse_args()

    # List cameras if requested
    if args.list_cameras or args.camera is None:
        available = list_available_cameras()
        if args.list_cameras:
            return
        if not available:
            print("\nNo cameras found. Please connect a USB webcam.")
            return
        if args.camera is None:
            print(f"\nNo camera specified. Using camera {available[0]}")
            args.camera = available[0]

    # Configuration
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cpu")
    num_joints = 6
    act_dim = 6

    print("\n" + "=" * 60)
    print("Vision-Based Policy Controller")
    print("=" * 60)

    # Initialize camera source
    print(f"\n[1/4] Initializing camera {args.camera}...")
    camera = CameraSource(camera_id=args.camera, show_preview=True)
    width, height = camera.get_resolution()
    print(f"      Camera resolution: {width}x{height}")

    # Initialize robot interface
    print(f"\n[2/4] Connecting to robot on {args.robot_port}...")
    robot_interface = So101RobotInterface(
        robot_id="so101", port=args.robot_port, urdf_path=args.urdf_path
    )

    # Initialize controller configuration
    controller_config = ControllerConfiguration(
        device=device,
        # joint_lower=[-2.8, -1.57, -2.35, -1.57, -3.14, 0.0],
        # joint_upper=[2.8, 1.57, 2.35, 1.57, 3.14, 1.0],
        # tolerance=torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    )

    # Initialize policy controller
    controller = PolicyController(
        controller_config=controller_config, robot_interface=robot_interface, beta=0.5
    )

    # Load the appropriate vision policy based on policy type
    print(
        f"\n[3/4] Loading {args.policy_type.upper()} vision policy from checkpoint..."
    )
    if args.policy_type == "fc":
        vision_policy = get_vision_fc_policy(
            policy_config=PolicyConfiguration(policy_path=args.checkpoint),
            device=device,
        )
        print(f"      Using FC policy (512-D features)")
    else:  # conv
        vision_policy = get_vision_conv_policy(
            policy_config=PolicyConfiguration(policy_path=args.checkpoint),
            device=device,
        )
        print(f"      Using Conv policy (1024-D spatial features)")

    # A favorable starting position for vision policy
    controller_config.hz = 60
    controller.run_policy(
        StaticPositionPolicy(
            torch.tensor(
                [
                    math.radians(0),
                    math.radians(0),
                    math.radians(-25),
                    math.radians(65),
                    math.radians(-50),
                    math.radians(0),
                ]
            ),
        ),
        terminate_on_steady_state=False,
    )

    # Run the vision-based policy
    print(f"\n[4/4] Starting policy execution...")
    print(f"      Control frequency: {controller_config.hz} Hz")
    print(f"      Device: {device}")
    print("\n" + "=" * 60)
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")

    try:
        controller_config.hz = 30
        controller.run_policy(
            policy=vision_policy, camera_source=camera, terminate_on_steady_state=False
        )
    finally:
        # Clean up
        camera.release()
        print("\n[INFO] Cleanup complete")

    try:
        controller_config.hz = 60
        controller.run_policy(
            get_safe_position_policy(), terminate_on_steady_state=False
        )
    finally:
        print("\n[INFO] Reached safe position. Exiting.")
        controller.robot_interface.robot.bus.disable_torque()


if __name__ == "__main__":
    main()

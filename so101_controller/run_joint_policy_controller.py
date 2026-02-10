#!/usr/bin/env python3
"""
Example script demonstrating how to use JointPositionPolicy with PolicyController.

This script shows how to:
1. Load a pretrained JointPositionPolicy checkpoint
2. Run the joint position-based policy on the robot
"""

import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from so101_controller.ControllerConfiguration import ControllerConfiguration
from so101_controller.JointPositionPolicy import JointPositionPolicy
from so101_controller.PolicyConfiguration import PolicyConfiguration
from so101_controller.PolicyController import PolicyController
from so101_controller.So101RobotInterface import So101RobotInterface
from so101_controller.StaticPositionPolicy import (
    StaticPositionPolicy,
    get_home_position_policy,
    get_safe_position_policy,
)
import torch
import argparse


def demo_routine(controller: PolicyController, joint_pos_policy: JointPositionPolicy):

    static_policy_1 = StaticPositionPolicy(
        torch.tensor(
            [
                math.radians(-91.88955996548749),
                math.radians(35.79920739762218),
                math.radians(87.5336322869955),
                math.radians(-92.35395189003437),
                math.radians(0.05564830272678023),
                math.radians(0.7920792079207921),
            ]
        )
    )

    static_policy_2 = StaticPositionPolicy(
        torch.tensor(
            [
                math.radians(88.17946505608282),
                math.radians(-5.239982386613832),
                math.radians(55.51569506726457),
                math.radians(88.8316151202749),
                math.radians(-0.38953811908736213),
                math.radians(0.7260726072607261),
            ]
        )
    )

    # Run policies
    controller.run_policy(get_home_position_policy(), terminate_on_steady_state=False)
    controller.run_policy(joint_pos_policy, terminate_on_steady_state=False)
    controller.run_policy(static_policy_1, terminate_on_steady_state=False)
    controller.run_policy(joint_pos_policy, terminate_on_steady_state=False)
    controller.run_policy(static_policy_2, terminate_on_steady_state=False)
    controller.run_policy(joint_pos_policy, terminate_on_steady_state=False)
    controller.run_policy(get_safe_position_policy(), terminate_on_steady_state=False)


def get_joint_position_policy(
    policy_config: PolicyConfiguration, device: torch.device
) -> JointPositionPolicy:
    # 1) Load exported checkpoint (container dict)
    ckpt = torch.load(policy_config.policy_path, map_location=device)

    # handle both “container dict” and “raw state_dict” cases
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        obs_dim = ckpt.get("obs_dim", policy_config.obs_dim)
        act_dim = ckpt.get("act_dim", policy_config.act_dim)
    else:
        state_dict = ckpt
        obs_dim = policy_config.obs_dim
        act_dim = policy_config.act_dim

    # 2) Build policy with matching dims
    policy = JointPositionPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Run joint position-based policy on SO-101 robot"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint file",
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
        default="so101_controller/assets/SO101/so101_new_calib.urdf",
        help="Path to robot URDF file",
    )
    args = parser.parse_args()

    # Configuration
    device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("Joint Position-Based Policy Controller")
    print("=" * 60)

    # Initialize robot interface
    print(f"\n[1/3] Connecting to robot on {args.robot_port}...")
    robot_interface = So101RobotInterface(
        robot_id="so101", port=args.robot_port, urdf_path=args.urdf_path
    )

    # Initialize controller configuration
    controller_config = ControllerConfiguration(device=device)

    # Initialize policy controller
    controller = PolicyController(
        controller_config=controller_config, robot_interface=robot_interface, beta=1.0
    )

    # Load the joint position policy
    print(f"\n[2/3] Loading joint position policy from checkpoint...")
    policy_config = PolicyConfiguration(policy_path=args.checkpoint)
    joint_pos_policy = get_joint_position_policy(policy_config, device)

    # Run the policy
    print(f"\n[3/3] Starting policy execution...")
    print(f"      Control frequency: {controller_config.hz} Hz")
    print(f"      Device: {device}")
    print("\n" + "=" * 60)
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")

    try:
        demo_routine(controller, joint_pos_policy)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Stopping...")
    finally:
        # Clean up
        robot_interface.robot.bus.disable_torque()
        print("\n[INFO] Cleanup complete")


if __name__ == "__main__":
    main()

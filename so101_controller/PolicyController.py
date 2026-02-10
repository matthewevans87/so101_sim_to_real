import torch
import torch.nn as nn
from so101_controller.ControllerConfiguration import ControllerConfiguration
from so101_controller.So101RobotInterface import So101RobotInterface
from so101_controller.StormVisionPolicy import VisionConvPolicy, VisionFcPolicy


import time
from typing import Callable, Optional


class PolicyController:
    def __init__(
        self,
        controller_config: ControllerConfiguration,
        robot_interface: So101RobotInterface,
        v_max: float = 1.0,
        beta: float = 0.5,
    ):
        self.robot_interface = robot_interface
        self.cfg = controller_config
        self.v_max = v_max  # max joint velocity (rad/s)
        self.beta = torch.clamp(
            torch.tensor([beta]), torch.tensor([0.0]), torch.tensor([1.0])
        )  # smoothing factor for velocity commands

    def run_policy(
        self,
        policy: nn.Module | Callable,
        camera_source: Optional[Callable] = None,
        terminate_on_steady_state: bool = False,
    ):
        """
        Run a policy on the robot.

        Args:
            policy: The policy to run (can be joint-based or vision-based)
            camera_source: Optional callable that returns camera RGB tensor (H, W, 3).
                          Required for StormVisionPolicy instances.
        """
        # Check if we're using a vision-based policy
        is_vision_policy = isinstance(policy, VisionConvPolicy) or isinstance(
            policy, VisionFcPolicy
        )

        if is_vision_policy and camera_source is None:
            raise ValueError("StormVisionPolicy requires a camera_source callable")

        dt = 1.0 / self.cfg.hz
        print(f"[PolicyController] Running at ~{self.cfg.hz} Hz. CTRL+C to stop.")
        if is_vision_policy:
            print("[PolicyController] Using vision-based policy (ResNet18 + joints)")
        else:
            print("[PolicyController] Using joint-based policy")

        joint_lower = torch.tensor(
            self.cfg.joint_lower, dtype=torch.float32, device=self.cfg.device
        )
        joint_upper = torch.tensor(
            self.cfg.joint_upper, dtype=torch.float32, device=self.cfg.device
        )

        done = False
        prev_q_target = None
        try:
            while not done:

                try:
                    # 1) Read current joint positions
                    q = self.robot_interface.get_joint_positions().to(
                        self.cfg.device
                    )  # [num_joints]
                except ConnectionError as e:
                    print(
                        f"[PolicyController] Warning: Failed to read joint positions: {e}. Skipping iteration."
                    )
                    time.sleep(dt)
                    continue

                # 2) Compute action based on policy type
                if is_vision_policy:
                    # Get camera image
                    assert camera_source is not None  # for type checker
                    camera_rgb = camera_source()  # Should return (H, W, 3) tensor
                    camera_rgb = camera_rgb.to(self.cfg.device)

                    # Vision policy takes camera + joint positions separately
                    a = policy(camera_rgb, q)
                else:
                    # Joint-based policy takes observation vector
                    obs = q.unsqueeze(0)  # [1, obs_dim]
                    a = policy(obs)[0]

                # 3) Map action from [-1, 1] to joint targets
                t = 0.5 * (a + 1.0)

                q_target = joint_lower + t * (joint_upper - joint_lower)

                # Apply velocity limits
                delta = (
                    q_target - prev_q_target
                    if prev_q_target is not None
                    else torch.zeros_like(q_target)
                )
                delta_max = self.v_max * dt
                delta_clamped = torch.clamp(delta, -delta_max, delta_max)
                q_target = (
                    prev_q_target if prev_q_target is not None else q
                ) + delta_clamped

                # Apply low-pass filtering
                if prev_q_target is not None:
                    q_target = (1 - self.beta) * prev_q_target + self.beta * q_target

                # 4) Send commands
                self.robot_interface.send_joint_positions(q_target)
                prev_q_target = q_target.clone()

                # Termination check based on closeness to target
                if terminate_on_steady_state:
                    done = torch.all(torch.abs(q - q_target) <= self.cfg.tolerance)
                else:
                    done = False

                # 5) Sleep
                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n[go_up_runner] Stopped by user.")

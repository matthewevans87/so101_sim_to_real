# ---------------------------
#  Robot interface wrapper
# ---------------------------


from so101_controller.constants import IN_PER_M
import torch
import numpy
import pinocchio as pin
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


import math
import os


class So101RobotInterface:
    """
    Thin adapter around LeRobot / SO-101 controls.
    """

    def __init__(
        self,
        robot_id: str = "so101",
        port: str = "/dev/ttyACM0",
        urdf_path: str = "so101_new_calib.urdf",
    ):
        self.robot_id = robot_id

        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        max_relative_target = {
            "shoulder_pan": 10.0,
            "shoulder_lift": 10.0,
            "elbow_flex": 10.0,
            "wrist_flex": 10.0,
            "wrist_roll": 10.0,
            "gripper": 100.0,
        }

        self.joint_static_override = {
            "shoulder_pan": None,
            "shoulder_lift": None,
            "elbow_flex": None,
            "wrist_flex": None,
            "wrist_roll": -53.0,  # degrees
            "gripper": None,
        }

        robot_cfg = SO101FollowerConfig(
            port=port, id=robot_id, max_relative_target=max_relative_target
        )
        self.robot = SO101Follower(robot_cfg)
        self.robot.connect()
        urdf_file = urdf_path
        urdf_dir = os.path.dirname(urdf_file)
        self.robot_model = pin.buildModelsFromUrdf(
            filename=urdf_file, package_dirs=urdf_dir
        )[0]
        self.robot_data = self.robot_model.createData()  # type: ignore
        ee_id = self.robot_model.getFrameId("gripper")
        self.oMf = self.robot_data.oMf[ee_id]  # type: ignore
        # self.SAFETY_MIN_HEIGHT = 0.039  # m
        self.SAFETY_MIN_HEIGHT = -0.05  # m

    def get_joint_positions(self) -> torch.Tensor:
        """Return current joint positions in radians as a 1D tensor [num_joints]."""
        observations = self.robot.get_observation()
        joint_positions = [
            math.radians(observations[f"{joint}.pos"]) for joint in self.joint_names
        ]
        return torch.tensor(joint_positions, dtype=torch.float32)

    def send_joint_positions(self, q_target: torch.Tensor):
        """Command joint positions (same order as cfg.dof_names)."""
        joint_positions = q_target.detach().cpu().tolist()

        # convert joint positions to degrees for LeRobot API
        joint_positions = [math.degrees(q) for q in joint_positions]

        # apply overrides
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in self.joint_static_override:
                static_value = self.joint_static_override[joint_name]
                if static_value is not None:
                    joint_positions[i] = static_value

        new_location_heights = [
            So101RobotInterface.compute_height(
                q_target.numpy(), self.robot_model, self.robot_data, j
            )
            for j in self.joint_names
        ]

        # print([lh for lh in new_location_heights])
        # print([lh * IN_PER_M for lh in new_location_heights])

        # if any(
        #     loc_height < self.SAFETY_MIN_HEIGHT for loc_height in new_location_heights
        # ):
        #     print(
        #         f"[safety] rejecting command: one or more tip heights below {self.SAFETY_MIN_HEIGHT:.3f} m"
        #     )
        #     return

        # Send joint commands one at a time for debugging more safely
        # for joint_name, joint_position in zip(self.joint_names, joint_positions):
        #     self.robot.send_action({f"{joint_name}.pos": joint_position})

        self.robot.send_action(
            {f"{joint}.pos": q for joint, q in zip(self.joint_names, joint_positions)}
        )

    def print_info(self):
        """Print current joint positions to console."""
        obs = self.robot.get_observation()
        print(f"[{self.robot_id}] Joint positions (deg): ", end="")
        print(obs)

        q = self.get_joint_positions()

        print(f"[{self.robot_id}] Joint heights (in): ", end="")

        current_location_heights = [
            So101RobotInterface.compute_height(
                q.numpy(), self.robot_model, self.robot_data, j
            )
            for j in self.joint_names
        ]

        joint_heights = {
            f"{joint_name}": joint_height * IN_PER_M
            for joint_name, joint_height in zip(
                self.joint_names, current_location_heights
            )
        }

        print(joint_heights)

    @staticmethod
    def compute_height(
        q: numpy.ndarray,
        model: pin.Model,  # type: ignore
        data: pin.Data,  # type: ignore
        joint_frame_name: str = "gripper",
        tip_offset_local: numpy.ndarray = numpy.array(
            [0.0, 0.0, 0.0]
        ),  # in gripper frame
    ) -> float:
        """
        Compute the tip height (z in world/base frame) given joint configuration q.

        Args:
            q: Joint configuration (shape: [model.nq,]) in radians.
            model: Pinocchio model for SO-101.
            data: Pinocchio data (same model).
            ee_frame_name: Name of the end-effector frame (e.g. "gripper").
            tip_offset_local: 3D vector of the tip position in the EE frame.

        Returns:
            Tip height (z coordinate) in the world/base frame (float).
        """
        assert q.shape[0] == model.nq, f"q has shape {q.shape}, expected ({model.nq},)"  # type: ignore

        # Forward kinematics
        pin.forwardKinematics(model, data, q)  # type: ignore
        pin.updateFramePlacements(model, data)  # type: ignore

        # Get end-effector frame transform oMf: world/base -> ee_frame
        ee_id = model.getFrameId(joint_frame_name)  # type: ignore
        oMf: pin.SE3 = data.oMf[ee_id]  # type: ignore

        # Tip position in world/base frame:
        # p_tip = oMf.translation + oMf.rotation @ tip_offset_local
        p_tip_world = oMf.translation + oMf.rotation @ tip_offset_local

        # Height is z-coordinate
        return float(p_tip_world[2])

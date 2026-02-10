from dataclasses import dataclass
from typing import Sequence


@dataclass
class PolicyConfiguration:
    # Path to exported policy checkpoint (plain PyTorch state_dict for GoUpPolicy)
    policy_path: str = "policy.pt"

    # Joint ordering must match your IsaacLab env cfg dof_names
    dof_names: Sequence[str] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    # Observation and action sizes
    obs_dim: int = 6  # [q]
    act_dim: int = 6  # one action per joint

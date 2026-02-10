import torch
from so101_controller.constants import SO101_JOINT_LIMITS_DEG


import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ControllerConfiguration:
    joint_lower: Sequence[float] = tuple(
        math.radians(l[0]) for l in SO101_JOINT_LIMITS_DEG
    )
    joint_upper: Sequence[float] = tuple(
        math.radians(l[1]) for l in SO101_JOINT_LIMITS_DEG
    )

    # Termination tolerance per joint (radians)
    tolerance: torch.Tensor = torch.tensor(
        [
            math.radians(1.0),
            math.radians(2.0),
            math.radians(3.0),
            math.radians(1.0),
            math.radians(1.0),
            math.radians(1.0),
        ]
    )

    # Control loop
    hz: float = 30  # control frequency (20–50 Hz is reasonable); we used 120 Hz in sim
    device: torch.device = torch.device("cpu")  # probably CPU on the robot

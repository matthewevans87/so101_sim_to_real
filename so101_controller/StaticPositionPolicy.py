import torch
from so101_controller.constants import SO101_JOINT_LIMITS_DEG


import math


class StaticPositionPolicy:
    def __init__(self, position: torch.Tensor):

        self.position = position
        self.joint_lower = torch.tensor(
            [math.radians(l[0]) for l in SO101_JOINT_LIMITS_DEG]
        )
        self.joint_upper = torch.tensor(
            [math.radians(l[1]) for l in SO101_JOINT_LIMITS_DEG]
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:

        home_q = self.position.unsqueeze(0)

        t = (home_q - self.joint_lower) / (self.joint_upper - self.joint_lower)
        a = 2.0 * t - 1.0
        return torch.clamp(a, -1.0, 1.0)


def get_home_position_policy() -> StaticPositionPolicy:
    return StaticPositionPolicy(
        torch.tensor(
            [
                math.radians(0),
                math.radians(0),
                math.radians(0),
                math.radians(0),
                math.radians(0),
                math.radians(0),
            ]
        )
    )


def get_safe_position_policy() -> StaticPositionPolicy:
    return StaticPositionPolicy(
        torch.tensor(
            [
                math.radians(-4.079003864319446),
                math.radians(-100.0),
                math.radians(98.44393592677346),
                math.radians(75.47660311958407),
                math.radians(-0.25604551920341123),
                math.radians(0.0),
            ]
        )
    )

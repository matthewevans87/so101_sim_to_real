# ---------------------------
#  Policy NN (deployment)
# ---------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F


class JointPositionPolicy(nn.Module):
    """Matches SKRL Gaussian policy backbone: 2x [32, ELU] → actions."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, act_dim)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(obs))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # PPO policy in Isaac Lab has no extra activation, but we exported
        # mean actions in [-1, 1], so tanh is fine / safe here.
        return torch.tanh(x)

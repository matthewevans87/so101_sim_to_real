"""policy.py — PolicyMLP definition and weight loading for real-robot inference.

This is the single source of truth for the MLP architecture at deploy time.
The same class definition lives in so101_rl/scripts/skrl/export_bundle.py —
if you change the architecture here, update it there too.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bundle import DeployBundle


class PolicyMLP(nn.Module):
    """Plain ELU-activated MLP with tanh output.

    Architecture auto-detected from the bundle manifest.
    """

    def __init__(self, obs_dim: int, hidden_dims: list[int], act_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def load_policy(bundle: DeployBundle, device: str | torch.device = "cpu") -> PolicyMLP:
    """Load policy weights from a deploy bundle.

    Parameters
    ----------
    bundle:
        Validated DeployBundle.
    device:
        Target device for inference (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns
    -------
    PolicyMLP
        Model in eval mode on the requested device.
    """
    payload = torch.load(bundle.policy_path, map_location="cpu", weights_only=True)

    obs_dim: int = int(payload["obs_dim"])
    act_dim: int = int(payload["act_dim"])
    hidden_dims: list[int] = [int(h) for h in payload["hidden_dims"]]

    if obs_dim != bundle.obs_dim:
        raise ValueError(
            f"policy.pt obs_dim={obs_dim} does not match manifest obs_dim={bundle.obs_dim}."
        )
    if act_dim != bundle.act_dim:
        raise ValueError(
            f"policy.pt act_dim={act_dim} does not match manifest act_dim={bundle.act_dim}."
        )

    model = PolicyMLP(obs_dim=obs_dim, hidden_dims=hidden_dims, act_dim=act_dim)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()

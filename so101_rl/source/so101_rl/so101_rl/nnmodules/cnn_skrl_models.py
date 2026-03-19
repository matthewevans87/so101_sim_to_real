"""skrl-compatible actor and critic models for the trainable CNN feature extractor.

These models are used when ``vision_encoder.type == "trainable_cnn"``; they are
constructed manually inside ``train.py`` (bypassing skrl's Runner / YAML model
instantiator) so that PPO's optimiser updates the CNN backbone.

Policy (actor) architecture
-----------------------------
    flat obs = [image_pixels (3×H×W), joint_positions (num_joints)]
    → split → image (N,3,H,W) → TrainableCnnFeatureExtractor → (N, cnn_output_dim)
              joints (N, num_joints) ──────────────────────────→ concat
    → ELU MLP head (head_hidden_dims) → action means (N, num_actions)
    + learnable log_std parameter

Value (critic) architecture
-----------------------------
    actor obs (N, 3*H*W + num_joints)  (same tensor skrl passes to the value)
    → slice last num_joints dims → joint positions (N, num_joints)
    → ELU MLP (hidden_dims) → scalar value (N, 1)

skrl's single-agent SequentialTrainer always passes the policy observation as
``states`` to the value function; it does not separately route the "critic"
privileged observations.  We therefore slice the proprioceptive tail (joint
positions) so the critic MLP remains tractable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from so101_utils.feature_extraction.feature_extraction import (
    TrainableCnnFeatureExtractor,
)


class CnnGaussianPolicy(GaussianMixin, Model):
    """Actor policy: trainable CNN backbone + joint-concat MLP head."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        # Image / CNN config
        image_height: int,
        image_width: int,
        num_joints: int,
        cnn_channels: list,
        cnn_kernel_sizes: list,
        cnn_strides: list,
        cnn_mlp_hidden_dims: list,
        cnn_output_dim: int,
        # Policy head
        head_hidden_dims: list,
        # GaussianMixin args
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
    ):
        if head_hidden_dims is None:
            raise ValueError(
                "head_hidden_dims must be provided explicitly. "
                "Set models.policy.head_dims in skrl_ppo_cfg.yaml."
            )
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self._image_h = image_height
        self._image_w = image_width
        self._num_joints = num_joints
        self._image_flat_dim = 3 * image_height * image_width

        self._cnn = TrainableCnnFeatureExtractor(
            in_channels=3,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            mlp_hidden_dims=cnn_mlp_hidden_dims,
            output_dim=cnn_output_dim,
        )

        # Policy head: CNN features concatenated with joint positions
        head_in_dim = cnn_output_dim + num_joints
        head_layers: list = []
        prev = head_in_dim
        for h in head_hidden_dims:
            head_layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        head_layers.append(nn.Linear(prev, self.num_actions))
        self._head = nn.Sequential(*head_layers)

        # Learnable log-std (shared across all envs in the batch)
        self.log_std_parameter = nn.Parameter(
            initial_log_std * torch.ones(self.num_actions)
        )

    def compute(self, inputs, role):
        obs = inputs["states"]  # (N, H*W*3 + num_joints)

        images_flat = obs[:, : self._image_flat_dim]
        joints = obs[:, self._image_flat_dim :]

        images = images_flat.view(-1, 3, self._image_h, self._image_w)  # (N, 3, H, W)
        cnn_feats = self._cnn(images)  # (N, cnn_output_dim)

        x = torch.cat([cnn_feats, joints], dim=-1)  # (N, cnn_output_dim + num_joints)
        mean = self._head(x)  # (N, num_actions)

        return mean, self.log_std_parameter, {}


class CnnDeterministicValue(DeterministicMixin, Model):
    """Critic value function: MLP on the proprioceptive tail of the actor observation.

    skrl's single-agent trainer passes the full policy observation as ``states``
    to the value function.  ``num_proprioception`` controls how many trailing
    dimensions of that vector are fed to the MLP (e.g. joint positions).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_proprioception: int | None = None,
        hidden_dims: list = None,
        clip_actions: bool = False,
    ):
        if hidden_dims is None:
            raise ValueError(
                "hidden_dims must be provided explicitly. "
                "Set models.value.hidden_dims in skrl_ppo_cfg.yaml."
            )
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        # MLP input: proprioceptive slice when provided, else full observation.
        mlp_in_dim = (
            num_proprioception
            if num_proprioception is not None
            else self.num_observations
        )
        self._num_proprio = num_proprioception

        layers: list = []
        prev = mlp_in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self._net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        states = inputs["states"]
        if self._num_proprio is not None:
            # Extract joint positions from the trailing dims of the actor obs.
            states = states[:, -self._num_proprio :]
        return self._net(states), {}


class MonitoredCnnPPO(PPO):
    """PPO subclass that logs CNN-specific diagnostics to TensorBoard.

    Adds five scalars under the ``CNN /`` prefix (visible in TensorBoard):

    Gradient flow (proves backprop reaches the CNN backbone):
        ``CNN / grad_norm_at_output``  — L2 norm of ∂loss/∂cnn_features,
                                        last mini-batch of each update cycle.
                                        Any non-zero value means gradients are
                                        flowing through the CNN.

    Weight diagnostics (proves the optimizer is updating CNN params):
        ``CNN / weight_norm``          — L2 norm of all CNN parameters.
        ``CNN / weight_delta``         — L2 change in CNN params per PPO update
                                        cycle (must be > 0 for learning to occur).

    Feature quality:
        ``CNN / feature_mean``         — Mean of CNN output features, last batch.
        ``CNN / feature_std``          — Std of CNN output features, last batch.
                                        Healthy range: std > 0.01 (not collapsed).
    """

    def __init__(self, cnn_module: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monitored_cnn = cnn_module
        self._cnn_grad_out_norm: float = 0.0
        self._cnn_feature_mean: float = 0.0
        self._cnn_feature_std: float = 0.0

        # Backward hook: fires every mini-batch backward pass.
        # Captures the gradient of the loss w.r.t. the CNN's output features;
        # a non-zero value proves end-to-end backprop reaches the CNN.
        def _bwd_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self._cnn_grad_out_norm = grad_output[0].detach().norm().item()

        self._monitored_cnn.register_full_backward_hook(_bwd_hook)

        # Forward hook: captures CNN output feature statistics on every forward pass.
        def _fwd_hook(module, inputs, output):
            out = output.detach()
            self._cnn_feature_mean = out.mean().item()
            self._cnn_feature_std = out.std().item()

        self._monitored_cnn.register_forward_hook(_fwd_hook)

    def _update(self, timestep: int, timesteps: int) -> None:
        # Flatten all CNN parameters to a single vector before the update so we
        # can compute the L2 weight change afterwards.
        with torch.no_grad():
            params_before = torch.cat(
                [p.data.view(-1) for p in self._monitored_cnn.parameters()]
            )

        super()._update(timestep, timesteps)

        with torch.no_grad():
            params_after = torch.cat(
                [p.data.view(-1) for p in self._monitored_cnn.parameters()]
            )
            weight_delta = (params_after - params_before).norm().item()
            weight_norm = params_after.norm().item()

        self.track_data("CNN / weight_norm", weight_norm)
        self.track_data("CNN / weight_delta", weight_delta)
        self.track_data("CNN / grad_norm_at_output", self._cnn_grad_out_norm)
        self.track_data("CNN / feature_mean", self._cnn_feature_mean)
        self.track_data("CNN / feature_std", self._cnn_feature_std)

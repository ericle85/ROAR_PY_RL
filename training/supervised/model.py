"""MLP policy network matching SB3's actor architecture."""

from typing import List

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """Initialize a linear layer with orthogonal weights and zero bias.

    This matches SB3's initialization pattern.

    Args:
        layer: Linear layer to initialize.
        gain: Gain for orthogonal initialization.

    Returns:
        The initialized layer.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLPPolicy(nn.Module):
    """MLP policy network matching SB3's MlpPolicy actor architecture.

    Architecture:
        Linear(obs_dim -> hidden) -> ReLU ->
        Linear(hidden -> hidden) -> ReLU ->
        Linear(hidden -> action_dim) -> Tanh

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_sizes: List of hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim: int = 49,
        action_dim: int = 2,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Build network layers
        layers = []
        in_dim = obs_dim

        for hidden_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output layer with smaller gain (matching SB3's policy head init)
        layers.append(layer_init(nn.Linear(in_dim, action_dim), gain=0.01))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observations tensor of shape (batch, obs_dim).

        Returns:
            Actions tensor of shape (batch, action_dim) bounded to [-1, 1].
        """
        return self.network(obs)

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_sizes,
        }

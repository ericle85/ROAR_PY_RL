"""Policy networks for behavioral cloning."""

from typing import List, Optional

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


def conv_layer_init(layer: nn.Conv2d, gain: float = np.sqrt(2)) -> nn.Conv2d:
    """Initialize a conv layer with orthogonal weights and zero bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
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
        obs_dim: int = 48,
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


class OccupancyEncoder(nn.Module):
    """CNN encoder for occupancy map.

    Takes a 50x50 occupancy map and outputs a compact embedding.

    Architecture:
        Conv2d(1, 32, 5, stride=2) -> ReLU -> 23x23
        Conv2d(32, 64, 3, stride=2) -> ReLU -> 11x11
        Conv2d(64, 64, 3, stride=2) -> ReLU -> 5x5
        Flatten -> 1600
        Linear(1600, embed_dim)
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            conv_layer_init(nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1)),
            nn.ReLU(),
            conv_layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            conv_layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size: 50 -> 24 -> 12 -> 6 = 6x6x64 = 2304
        # Actually let's compute it properly
        self._compute_conv_output_size()

        self.fc = layer_init(nn.Linear(self._conv_output_size, embed_dim))

    def _compute_conv_output_size(self):
        """Compute the output size of the conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 50, 50)
            out = self.conv(dummy)
            self._conv_output_size = out.shape[1]

    def forward(self, occupancy_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            occupancy_map: Shape (batch, 50, 50) or (batch, 1, 50, 50)

        Returns:
            Embedding of shape (batch, embed_dim)
        """
        if occupancy_map.dim() == 3:
            occupancy_map = occupancy_map.unsqueeze(1)  # Add channel dim

        # Normalize to [0, 1] if needed (occupancy map is 0-255)
        if occupancy_map.max() > 1.0:
            occupancy_map = occupancy_map / 255.0

        x = self.conv(occupancy_map)
        x = self.fc(x)
        return x


class CNNMLPPolicy(nn.Module):
    """CNN+MLP policy for occupancy map + vector observations.

    Architecture:
        Occupancy Map (50x50) -> CNN Encoder -> 64-dim embedding
        Vector Obs (48-dim) -> concat with embedding -> 112-dim
        MLP: 112 -> 256 -> 256 -> 2 (action)

    Args:
        vector_obs_dim: Dimension of non-image observations (default 48).
        action_dim: Action dimension (default 2).
        hidden_sizes: MLP hidden layer sizes.
        occupancy_embed_dim: CNN output embedding dimension.
    """

    def __init__(
        self,
        vector_obs_dim: int = 48,
        action_dim: int = 2,
        hidden_sizes: List[int] = None,
        occupancy_embed_dim: int = 64,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.vector_obs_dim = vector_obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.occupancy_embed_dim = occupancy_embed_dim

        # CNN encoder for occupancy map
        self.occupancy_encoder = OccupancyEncoder(embed_dim=occupancy_embed_dim)

        # MLP for combined features
        combined_dim = vector_obs_dim + occupancy_embed_dim

        layers = []
        in_dim = combined_dim

        for hidden_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output layer
        layers.append(layer_init(nn.Linear(in_dim, action_dim), gain=0.01))
        layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        occupancy_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Can be called in two ways:
        1. forward(flattened_obs) - obs is (batch, vector_obs_dim + 2500)
        2. forward(vector_obs, occupancy_map) - separate inputs

        Args:
            obs: Either flattened observation (batch, 2548) or vector obs (batch, 48).
            occupancy_map: Optional occupancy map (batch, 50, 50) if obs is vector only.

        Returns:
            Actions of shape (batch, action_dim) bounded to [-1, 1].
        """
        if occupancy_map is None:
            # Flattened input - split it
            vector_obs = obs[:, :self.vector_obs_dim]
            occupancy_flat = obs[:, self.vector_obs_dim:]
            occupancy_map = occupancy_flat.view(-1, 50, 50)
        else:
            vector_obs = obs

        # Encode occupancy map
        occ_embedding = self.occupancy_encoder(occupancy_map)

        # Concatenate with vector observations
        combined = torch.cat([vector_obs, occ_embedding], dim=-1)

        # MLP forward
        return self.mlp(combined)

    def forward_split(
        self,
        vector_obs: torch.Tensor,
        occupancy_map: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with separate inputs (explicit version).

        Args:
            vector_obs: Vector observations of shape (batch, vector_obs_dim).
            occupancy_map: Occupancy map of shape (batch, 50, 50).

        Returns:
            Actions of shape (batch, action_dim) bounded to [-1, 1].
        """
        return self.forward(vector_obs, occupancy_map)

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "vector_obs_dim": self.vector_obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_sizes,
            "occupancy_embed_dim": self.occupancy_embed_dim,
        }

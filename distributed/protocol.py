"""
Protocol for rollout serialization between workers and learner.

Uses msgpack + msgpack-numpy for fast numpy array serialization.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import msgpack
    import msgpack_numpy as m
    # Patch msgpack for numpy support
    m.patch()
except ImportError as e:
    raise ImportError(
        "msgpack and msgpack-numpy are required. "
        "Install with: pip install msgpack msgpack-numpy"
    ) from e


@dataclass
class RolloutBatch:
    """
    A batch of rollout data collected by a worker.

    Contains all data needed for PPO training:
    - observations: states seen by the agent
    - actions: actions taken (discrete indices)
    - rewards: rewards received
    - dones: episode termination flags
    - values: value function estimates
    - log_probs: log probabilities of actions under policy
    - episode_starts: flags indicating episode boundaries
    """

    worker_id: int
    observations: np.ndarray  # (n_steps, obs_dim)
    actions: np.ndarray  # (n_steps,) discrete action indices
    rewards: np.ndarray  # (n_steps,)
    dones: np.ndarray  # (n_steps,)
    values: np.ndarray  # (n_steps,)
    log_probs: np.ndarray  # (n_steps,)
    episode_starts: np.ndarray  # (n_steps,) bool
    last_values: np.ndarray  # (1,) value estimate for last state
    last_dones: np.ndarray  # (1,) done flag for last state

    def __post_init__(self):
        """Validate shapes after initialization."""
        n_steps = len(self.observations)
        assert len(self.actions) == n_steps, f"actions shape mismatch: {len(self.actions)} vs {n_steps}"
        assert len(self.rewards) == n_steps, f"rewards shape mismatch: {len(self.rewards)} vs {n_steps}"
        assert len(self.dones) == n_steps, f"dones shape mismatch: {len(self.dones)} vs {n_steps}"
        assert len(self.values) == n_steps, f"values shape mismatch: {len(self.values)} vs {n_steps}"
        assert len(self.log_probs) == n_steps, f"log_probs shape mismatch: {len(self.log_probs)} vs {n_steps}"
        assert len(self.episode_starts) == n_steps, f"episode_starts shape mismatch: {len(self.episode_starts)} vs {n_steps}"

    @property
    def n_steps(self) -> int:
        """Number of timesteps in this rollout."""
        return len(self.observations)

    def serialize(self) -> bytes:
        """
        Serialize rollout batch to bytes for Redis transmission.

        Returns:
            bytes: msgpack-encoded rollout data
        """
        data = {
            "worker_id": self.worker_id,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "log_probs": self.log_probs,
            "episode_starts": self.episode_starts,
            "last_values": self.last_values,
            "last_dones": self.last_dones,
        }
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "RolloutBatch":
        """
        Deserialize rollout batch from bytes.

        Args:
            data: msgpack-encoded rollout data

        Returns:
            RolloutBatch instance
        """
        unpacked = msgpack.unpackb(data, raw=False)
        return cls(
            worker_id=unpacked["worker_id"],
            observations=unpacked["observations"],
            actions=unpacked["actions"],
            rewards=unpacked["rewards"],
            dones=unpacked["dones"],
            values=unpacked["values"],
            log_probs=unpacked["log_probs"],
            episode_starts=unpacked["episode_starts"],
            last_values=unpacked["last_values"],
            last_dones=unpacked["last_dones"],
        )


@dataclass
class PolicyWeights:
    """
    Policy weights for broadcasting to workers.

    Contains the state dict of the policy network serialized for transmission.
    """

    version: int  # Monotonically increasing version number
    state_dict: dict  # Model state dict with numpy arrays

    def serialize(self) -> bytes:
        """
        Serialize policy weights to bytes.

        Returns:
            bytes: msgpack-encoded policy weights
        """
        # Convert torch tensors to numpy if needed
        numpy_state_dict = {}
        for key, value in self.state_dict.items():
            if hasattr(value, "cpu"):  # torch tensor
                numpy_state_dict[key] = value.cpu().numpy()
            else:
                numpy_state_dict[key] = value

        data = {
            "version": self.version,
            "state_dict": numpy_state_dict,
        }
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "PolicyWeights":
        """
        Deserialize policy weights from bytes.

        Args:
            data: msgpack-encoded policy weights

        Returns:
            PolicyWeights instance
        """
        unpacked = msgpack.unpackb(data, raw=False)
        return cls(
            version=unpacked["version"],
            state_dict=unpacked["state_dict"],
        )


def test_protocol():
    """Test serialization roundtrip."""
    # Create test rollout
    n_steps = 100
    obs_dim = 64

    rollout = RolloutBatch(
        worker_id=0,
        observations=np.random.randn(n_steps, obs_dim).astype(np.float32),
        actions=np.random.randint(0, 9, size=(n_steps,)),
        rewards=np.random.randn(n_steps).astype(np.float32),
        dones=np.zeros(n_steps, dtype=bool),
        values=np.random.randn(n_steps).astype(np.float32),
        log_probs=np.random.randn(n_steps).astype(np.float32),
        episode_starts=np.zeros(n_steps, dtype=bool),
        last_values=np.array([0.0], dtype=np.float32),
        last_dones=np.array([False]),
    )
    rollout.episode_starts[0] = True

    # Serialize and deserialize
    serialized = rollout.serialize()
    print(f"Serialized size: {len(serialized)} bytes")

    deserialized = RolloutBatch.deserialize(serialized)

    # Verify
    assert deserialized.worker_id == rollout.worker_id
    assert np.allclose(deserialized.observations, rollout.observations)
    assert np.array_equal(deserialized.actions, rollout.actions)
    assert np.allclose(deserialized.rewards, rollout.rewards)
    assert np.array_equal(deserialized.dones, rollout.dones)
    assert np.allclose(deserialized.values, rollout.values)
    assert np.allclose(deserialized.log_probs, rollout.log_probs)
    assert np.array_equal(deserialized.episode_starts, rollout.episode_starts)

    print("Protocol test passed!")

    # Test policy weights
    weights = PolicyWeights(
        version=1,
        state_dict={
            "layer1.weight": np.random.randn(64, 32).astype(np.float32),
            "layer1.bias": np.random.randn(64).astype(np.float32),
        },
    )

    serialized_weights = weights.serialize()
    print(f"Serialized weights size: {len(serialized_weights)} bytes")

    deserialized_weights = PolicyWeights.deserialize(serialized_weights)
    assert deserialized_weights.version == weights.version
    assert np.allclose(
        deserialized_weights.state_dict["layer1.weight"],
        weights.state_dict["layer1.weight"],
    )

    print("Policy weights test passed!")


if __name__ == "__main__":
    test_protocol()

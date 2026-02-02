"""
Distributed PPO Learner (Python 3.11 + CUDA)

Receives rollouts from workers via Redis, trains PPO on GPU,
and broadcasts updated policy weights to workers.

Usage:
    python learner.py --num-workers 2 --total-timesteps 10000000
"""

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path for imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from distributed.config import Config
from distributed.protocol import RolloutBatch, PolicyWeights
from distributed.redis_client import RolloutQueue, PolicyChannel, RedisHealthCheck


logger = logging.getLogger(__name__)


def find_latest_model(root_path: Path) -> Optional[Path]:
    """
    Find the latest model checkpoint in a directory.

    Compatible with the original train_online.py checkpoint structure:
    models/{run_name}/logs/rl_model_XXXXX_steps.zip

    Args:
        root_path: Path to model directory (e.g., models/PPO_Discrete_RacingLine)

    Returns:
        Path to latest checkpoint, or None if not found
    """
    logs_path = root_path / "logs"
    if not logs_path.exists():
        # Try root_path directly (for distributed checkpoints)
        logs_path = root_path

    if not logs_path.exists():
        logger.warning(f"No model directory found at {root_path}")
        return None

    # Find all .zip files
    model_files = list(logs_path.glob("*.zip"))
    if not model_files:
        logger.warning(f"No model files found in {logs_path}")
        return None

    # Try to extract step numbers from filenames
    # Format: rl_model_XXXXX_steps.zip or ppo_step_XXXXX.zip
    def extract_steps(path: Path) -> int:
        name = path.stem
        parts = name.split("_")
        for i, part in enumerate(parts):
            if part.isdigit():
                return int(part)
            # Handle "stepXXXXX" format
            if part.startswith("step") and part[4:].isdigit():
                return int(part[4:])
        return 0

    # Sort by step number and return latest
    model_files.sort(key=extract_steps)
    latest = model_files[-1]
    logger.info(f"Found latest model: {latest}")
    return latest


class DummyEnv(gym.Env):
    """
    Dummy environment for learner that doesn't actually interact with CARLA.

    The learner only needs the observation and action spaces to initialize
    the PPO model. All actual data comes from workers via Redis.
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, False, False, {}


@dataclass
class LearnerStats:
    """Statistics tracking for the learner."""

    total_timesteps: int = 0
    total_updates: int = 0
    policy_version: int = 0
    rollouts_received: int = 0
    rollouts_from_workers: Dict[int, int] = None

    # Per-update stats
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_var: float = 0.0

    # Episode stats from workers
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0

    def __post_init__(self):
        if self.rollouts_from_workers is None:
            self.rollouts_from_workers = {}

    def log_rollout(self, rollout: RolloutBatch) -> None:
        """Log a received rollout."""
        self.rollouts_received += 1
        worker_id = rollout.worker_id
        self.rollouts_from_workers[worker_id] = self.rollouts_from_workers.get(worker_id, 0) + 1


class DistributedRolloutBuffer:
    """
    Buffer that collects rollouts from multiple workers.

    Aggregates data into a format compatible with PPO training.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        device: th.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize buffer.

        Args:
            buffer_size: Maximum number of steps to store
            obs_dim: Observation dimension
            device: Torch device for training
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.episode_starts = np.zeros(buffer_size, dtype=bool)

        # GAE outputs
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

        # Track last values/dones for GAE computation
        self._pending_last_values: List[np.ndarray] = []
        self._pending_last_dones: List[np.ndarray] = []
        self._pending_boundaries: List[int] = []

    def add_rollout(self, rollout: RolloutBatch) -> bool:
        """
        Add a rollout to the buffer.

        Args:
            rollout: RolloutBatch from worker

        Returns:
            bool: True if buffer is now full
        """
        n_steps = rollout.n_steps
        space_left = self.buffer_size - self.pos

        if n_steps > space_left:
            # Buffer would overflow - mark as full
            self.full = True
            logger.warning(f"Buffer overflow: {n_steps} steps, {space_left} space left")
            return True

        # Copy data to buffer
        end_pos = self.pos + n_steps
        self.observations[self.pos:end_pos] = rollout.observations
        self.actions[self.pos:end_pos] = rollout.actions
        self.rewards[self.pos:end_pos] = rollout.rewards
        self.dones[self.pos:end_pos] = rollout.dones
        self.values[self.pos:end_pos] = rollout.values
        self.log_probs[self.pos:end_pos] = rollout.log_probs
        self.episode_starts[self.pos:end_pos] = rollout.episode_starts

        # Track for GAE computation
        self._pending_last_values.append(rollout.last_values)
        self._pending_last_dones.append(rollout.last_dones)
        self._pending_boundaries.append(end_pos)

        self.pos = end_pos

        # Check if full
        if self.pos >= self.buffer_size:
            self.full = True

        return self.full

    def compute_returns_and_advantage(self) -> None:
        """
        Compute GAE advantages and returns.

        Must be called after buffer is full and before get_samples().
        """
        # We need to compute GAE for each segment separately
        # Each segment corresponds to a rollout from a worker

        start_pos = 0
        for i, end_pos in enumerate(self._pending_boundaries):
            last_values = self._pending_last_values[i]
            last_dones = self._pending_last_dones[i]

            n_steps = end_pos - start_pos
            last_gae_lam = 0.0

            for step in reversed(range(n_steps)):
                buf_idx = start_pos + step

                if step == n_steps - 1:
                    # Last step - use bootstrap value
                    next_non_terminal = 1.0 - float(last_dones[0])
                    next_values = last_values[0]
                else:
                    next_non_terminal = 1.0 - float(self.dones[buf_idx + 1])
                    next_values = self.values[buf_idx + 1]

                delta = (
                    self.rewards[buf_idx]
                    + self.gamma * next_values * next_non_terminal
                    - self.values[buf_idx]
                )
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[buf_idx] = last_gae_lam

            start_pos = end_pos

        # Returns = advantages + values
        self.returns = self.advantages + self.values

    def get_samples(self, batch_size: int):
        """
        Generator that yields random minibatches.

        Args:
            batch_size: Size of each minibatch

        Yields:
            Dictionary with tensors for training
        """
        indices = np.random.permutation(self.pos)

        for start_idx in range(0, self.pos, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield {
                "observations": th.as_tensor(self.observations[batch_indices]).to(self.device),
                "actions": th.as_tensor(self.actions[batch_indices]).to(self.device),
                "old_values": th.as_tensor(self.values[batch_indices]).to(self.device),
                "old_log_probs": th.as_tensor(self.log_probs[batch_indices]).to(self.device),
                "advantages": th.as_tensor(self.advantages[batch_indices]).to(self.device),
                "returns": th.as_tensor(self.returns[batch_indices]).to(self.device),
            }

    def reset(self) -> None:
        """Reset buffer for next collection cycle."""
        self.pos = 0
        self.full = False
        self._pending_last_values = []
        self._pending_last_dones = []
        self._pending_boundaries = []


class Learner:
    """
    Distributed PPO learner.

    Receives rollouts from workers, trains the policy on GPU,
    and broadcasts updated weights.
    """

    def __init__(
        self,
        config: Config,
        num_workers: int,
        obs_dim: int = None,  # Auto-detect from first rollout if None
        n_actions: int = 9,  # Default discrete actions
    ):
        """
        Initialize learner.

        Args:
            config: Configuration object
            num_workers: Expected number of workers
            obs_dim: Observation dimension (auto-detected if None)
            n_actions: Number of discrete actions
        """
        self.config = config
        self.num_workers = num_workers
        self.obs_dim = obs_dim  # Will be set from first rollout if None
        self.n_actions = n_actions
        self._model_initialized = False

        # Determine device
        if th.cuda.is_available():
            self.device = th.device("cuda")
            logger.info(f"Using CUDA device: {th.cuda.get_device_name()}")
        else:
            self.device = th.device("cpu")
            logger.warning("CUDA not available, using CPU")

        # Components (initialized in start())
        self.model: Optional[PPO] = None
        self.buffer: Optional[DistributedRolloutBuffer] = None
        self.rollout_queue: Optional[RolloutQueue] = None
        self.policy_channel: Optional[PolicyChannel] = None

        # State
        self.stats = LearnerStats()
        self._running = False

        # Wandb
        self.wandb_run = None

    def _initialize_model(self) -> PPO:
        """Initialize PPO model."""
        logger.info("Initializing PPO model...")

        # Create dummy env for model initialization
        env = DummyEnv(self.obs_dim, self.n_actions)
        env = DummyVecEnv([lambda: env])

        # Create model
        ppo_params = self.config.get_ppo_params()
        ppo_params["device"] = self.device

        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=None,  # We use wandb instead
            **ppo_params,
        )

        logger.info(f"Model initialized on {self.device}")
        return model

    def _get_policy_weights(self) -> PolicyWeights:
        """Get current policy weights for broadcasting."""
        state_dict = self.model.policy.state_dict()
        return PolicyWeights(
            version=self.stats.policy_version,
            state_dict=state_dict,
        )

    def _broadcast_policy(self) -> None:
        """Broadcast current policy weights to workers."""
        weights = self._get_policy_weights()
        num_subscribers = self.policy_channel.publish(weights)
        logger.info(f"Broadcast policy v{self.stats.policy_version} to {num_subscribers} subscribers")

    def start(self) -> bool:
        """
        Start the learner.

        Returns:
            bool: True if started successfully
        """
        logger.info("Starting learner...")

        # Check Redis
        health = RedisHealthCheck(self.config)
        if not health.wait_for_redis(timeout=30.0):
            logger.error("Redis not available")
            return False

        # Initialize Redis clients
        self.rollout_queue = RolloutQueue(self.config)
        self.policy_channel = PolicyChannel(self.config)

        # Clear old data
        self.rollout_queue.clear()
        self.policy_channel.clear()

        # Initialize model
        self.model = self._initialize_model()

        # Initialize buffer (size = n_steps * num_workers for full sync)
        buffer_size = self.config.n_steps * self.num_workers
        self.buffer = DistributedRolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Initialize wandb
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=f"learner_{self.num_workers}workers",
            config={
                "num_workers": self.num_workers,
                **self.config.get_ppo_params(),
            },
        )

        # Broadcast initial policy
        self.stats.policy_version = 1
        self._broadcast_policy()

        self._running = True
        logger.info("Learner started successfully")
        return True

    def stop(self) -> None:
        """Stop the learner and clean up."""
        logger.info("Stopping learner...")
        self._running = False

        # Save final checkpoint
        if self.model is not None:
            self._save_checkpoint("final")

        if self.wandb_run is not None:
            self.wandb_run.finish()

        logger.info("Learner stopped")

    def _save_checkpoint(self, name: str = None) -> None:
        """Save model checkpoint."""
        if name is None:
            name = f"step_{self.stats.total_timesteps}"

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f"ppo_{name}.zip"
        self.model.save(path)
        logger.info(f"Saved checkpoint: {path}")

    def _train_step(self) -> Dict[str, float]:
        """
        Perform one PPO training step.

        Returns:
            Dict with training metrics
        """
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantage()

        # Get training parameters
        clip_range = self.config.clip_range
        entropy_coef = self.config.ent_coef
        vf_coef = self.config.vf_coef
        max_grad_norm = self.config.max_grad_norm
        n_epochs = self.config.n_epochs
        batch_size = self.config.batch_size

        # Training metrics accumulators
        pg_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []

        # Train for n_epochs
        for epoch in range(n_epochs):
            for batch in self.buffer.get_samples(batch_size):
                observations = batch["observations"]
                actions = batch["actions"]
                old_values = batch["old_values"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy outputs
                values, log_probs, entropy = self.model.policy.evaluate_actions(
                    observations, actions
                )
                values = values.flatten()

                # Policy loss (clipped surrogate objective)
                ratio = th.exp(log_probs - old_log_probs)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss (optionally clipped)
                if self.config.clip_range_vf is not None:
                    values_clipped = old_values + th.clamp(
                        values - old_values, -self.config.clip_range_vf, self.config.clip_range_vf
                    )
                    value_loss_1 = (values - returns) ** 2
                    value_loss_2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * th.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_probs)
                else:
                    entropy_loss = -th.mean(entropy)

                # Total loss
                loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

                # Optimization step
                self.model.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.model.policy.parameters(), max_grad_norm)
                self.model.policy.optimizer.step()

                # Track metrics
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                with th.no_grad():
                    approx_kl = ((ratio - 1) - th.log(ratio)).mean().item()
                    clip_frac = (th.abs(ratio - 1) > clip_range).float().mean().item()
                    approx_kls.append(approx_kl)
                    clip_fractions.append(clip_frac)

        # Compute explained variance
        y_pred = self.buffer.values[:self.buffer.pos]
        y_true = self.buffer.returns[:self.buffer.pos]
        explained_var = explained_variance(y_pred, y_true)

        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
            "explained_variance": explained_var,
        }

    def run(self, total_timesteps: int = None) -> None:
        """
        Main training loop.

        Args:
            total_timesteps: Total timesteps to train (uses config if None)
        """
        if not self._running:
            raise RuntimeError("Learner not started")

        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        while self._running and self.stats.total_timesteps < total_timesteps:
            try:
                # Collect rollouts until buffer is full
                while not self.buffer.full:
                    # Pop rollout (blocking)
                    rollout = self.rollout_queue.pop(timeout=self.config.learner_pop_timeout)

                    if rollout is None:
                        logger.warning("Timeout waiting for rollout")
                        continue

                    self.buffer.add_rollout(rollout)
                    self.stats.log_rollout(rollout)
                    self.stats.total_timesteps += rollout.n_steps

                    logger.debug(
                        f"Received rollout from worker {rollout.worker_id}, "
                        f"buffer: {self.buffer.pos}/{self.buffer.buffer_size}"
                    )

                # Train
                logger.info(f"Training update {self.stats.total_updates + 1}...")
                train_start = time.time()

                metrics = self._train_step()

                train_time = time.time() - train_start

                # Update stats
                self.stats.total_updates += 1
                self.stats.policy_version += 1
                self.stats.policy_loss = metrics["policy_loss"]
                self.stats.value_loss = metrics["value_loss"]
                self.stats.entropy_loss = metrics["entropy_loss"]
                self.stats.approx_kl = metrics["approx_kl"]
                self.stats.clip_fraction = metrics["clip_fraction"]
                self.stats.explained_var = metrics["explained_variance"]

                # Broadcast updated policy
                self._broadcast_policy()

                # Log to wandb
                wandb.log({
                    "train/timesteps": self.stats.total_timesteps,
                    "train/updates": self.stats.total_updates,
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy_loss": metrics["entropy_loss"],
                    "train/approx_kl": metrics["approx_kl"],
                    "train/clip_fraction": metrics["clip_fraction"],
                    "train/explained_variance": metrics["explained_variance"],
                    "train/train_time": train_time,
                    "system/queue_length": self.rollout_queue.length(),
                    "system/policy_version": self.stats.policy_version,
                })

                # Log progress
                logger.info(
                    f"Update {self.stats.total_updates}: "
                    f"timesteps={self.stats.total_timesteps}, "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"kl={metrics['approx_kl']:.4f}, "
                    f"train_time={train_time:.1f}s"
                )

                # Reset buffer
                self.buffer.reset()

                # Checkpoint
                if self.stats.total_updates % (self.config.model_save_freq // self.config.n_steps) == 0:
                    self._save_checkpoint()

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break

            except Exception as e:
                logger.error(f"Error in training loop: {e}", exc_info=True)
                continue

        self.stop()


def main():
    """Main entry point for learner."""
    parser = argparse.ArgumentParser(description="Distributed PPO Learner")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to expect",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis server host",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port",
    )
    parser.add_argument(
        "--redis-password",
        type=str,
        default=None,
        help="Redis server password (if required)",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=94,
        help="Observation dimension",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=9,
        help="Number of discrete actions",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to specific checkpoint file to resume from",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to models directory (e.g., models/PPO_Discrete_RacingLine) - auto-finds latest",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [Learner] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create config
    config = Config(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=args.redis_password,
        total_timesteps=args.total_timesteps,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create learner
    learner = Learner(
        config=config,
        num_workers=args.num_workers,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
    )

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        learner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start and run
    if learner.start():
        # Determine checkpoint to load
        checkpoint_path = None
        if args.resume is not None:
            checkpoint_path = Path(args.resume)
        elif args.models_dir is not None:
            checkpoint_path = find_latest_model(Path(args.models_dir))

        # Load checkpoint if available
        if checkpoint_path is not None and checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            learner.model = PPO.load(checkpoint_path, device=learner.device)
            learner._broadcast_policy()
            logger.info("Checkpoint loaded and broadcast to workers")
        elif checkpoint_path is not None:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

        learner.run()
    else:
        logger.error("Failed to start learner")
        sys.exit(1)


if __name__ == "__main__":
    main()

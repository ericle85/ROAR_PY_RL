"""
Distributed PPO Learner (Python 3.11 + CUDA)

Receives rollouts from workers via Redis, trains PPO on GPU,
and broadcasts updated policy weights to workers.

Uses SB3's PPO infrastructure directly for training.

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
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch as th
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
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
        logs_path = root_path

    if not logs_path.exists():
        logger.warning(f"No model directory found at {root_path}")
        return None

    model_files = list(logs_path.glob("*.zip"))
    if not model_files:
        logger.warning(f"No model files found in {logs_path}")
        return None

    def extract_steps(path: Path) -> int:
        name = path.stem
        parts = name.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
            if part.startswith("step") and part[4:].isdigit():
                return int(part[4:])
        return 0

    model_files.sort(key=extract_steps)
    latest = model_files[-1]
    logger.info(f"Found latest model: {latest}")
    return latest


class DummyVecEnvWrapper(DummyVecEnv):
    """
    Wrapper to create a VecEnv compatible with SB3 from observation/action spaces.
    """

    def __init__(self, obs_dim: int, n_actions: int):
        def make_env():
            env = gym.Env()
            env.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            env.action_space = gym.spaces.Discrete(n_actions)
            env.reset = lambda **kwargs: (np.zeros(obs_dim, dtype=np.float32), {})
            env.step = lambda a: (np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {})
            return env

        super().__init__([make_env])


@dataclass
class LearnerStats:
    """Statistics tracking for the learner."""

    total_timesteps: int = 0
    total_updates: int = 0
    policy_version: int = 0
    rollouts_received: int = 0
    rollouts_from_workers: Dict[int, int] = None

    def __post_init__(self):
        if self.rollouts_from_workers is None:
            self.rollouts_from_workers = {}

    def log_rollout(self, rollout: RolloutBatch) -> None:
        """Log a received rollout."""
        self.rollouts_received += 1
        worker_id = rollout.worker_id
        self.rollouts_from_workers[worker_id] = self.rollouts_from_workers.get(worker_id, 0) + 1


class Learner:
    """
    Distributed PPO learner using SB3's infrastructure.

    Receives rollouts from workers, populates SB3's RolloutBuffer,
    and calls PPO.train() for updates.
    """

    def __init__(
        self,
        config: Config,
        num_workers: int,
        obs_dim: int = None,
        n_actions: int = 9,
    ):
        self.config = config
        self.num_workers = num_workers
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        if th.cuda.is_available():
            self.device = th.device("cuda")
            logger.info(f"Using CUDA device: {th.cuda.get_device_name()}")
        else:
            self.device = th.device("cpu")
            logger.warning("CUDA not available, using CPU")

        self.model: Optional[PPO] = None
        self.rollout_queue: Optional[RolloutQueue] = None
        self.policy_channel: Optional[PolicyChannel] = None

        self.stats = LearnerStats()
        self._running = False
        self.wandb_run = None

    def _create_model(self) -> PPO:
        """Create PPO model with SB3."""
        logger.info(f"Creating PPO model (obs_dim={self.obs_dim}, n_actions={self.n_actions})...")

        # Create dummy vec env for model initialization
        env = DummyVecEnvWrapper(self.obs_dim, self.n_actions)

        model = PPO(
            "MlpPolicy",
            env,
            **self.config.get_ppo_params(),
            device=self.device,
        )

        logger.info(f"Model created on {self.device}")
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

    def _populate_rollout_buffer(self, rollouts: List[RolloutBatch]) -> None:
        """
        Populate SB3's RolloutBuffer with collected rollouts.

        This mimics what PPO.collect_rollouts() does, but with external data.
        """
        buffer = self.model.rollout_buffer
        buffer.reset()

        for rollout in rollouts:
            for step in range(rollout.n_steps):
                # SB3 expects (n_envs, ...) shape, we have single env per worker
                obs = rollout.observations[step:step + 1]
                action = rollout.actions[step:step + 1]
                reward = rollout.rewards[step:step + 1]
                episode_start = rollout.episode_starts[step:step + 1]
                value = th.tensor([rollout.values[step]])
                log_prob = th.tensor([rollout.log_probs[step]])

                buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    episode_start=episode_start,
                    value=value,
                    log_prob=log_prob,
                )

        # Compute returns and advantages using last values from rollouts
        # Use the last rollout's final value for bootstrapping
        last_rollout = rollouts[-1]
        last_values = th.tensor(last_rollout.last_values).to(self.device)
        last_dones = th.tensor(last_rollout.last_dones.astype(np.float32)).to(self.device)

        buffer.compute_returns_and_advantage(last_values=last_values, dones=last_dones)

    def start(self) -> bool:
        """Start the learner."""
        logger.info("Starting learner...")

        health = RedisHealthCheck(self.config)
        if not health.wait_for_redis(timeout=30.0):
            logger.error("Redis not available")
            return False

        self.rollout_queue = RolloutQueue(self.config)
        self.policy_channel = PolicyChannel(self.config)

        # Clear old data
        self.rollout_queue.clear()
        self.policy_channel.clear()

        # Model will be created after we know obs_dim (from first rollout if not specified)
        if self.obs_dim is not None:
            self.model = self._create_model()

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=f"distributed_{self.num_workers}workers",
            config={
                "num_workers": self.num_workers,
                "obs_dim": self.obs_dim,
                **self.config.get_ppo_params(),
            },
        )

        self._running = True
        logger.info("Learner started successfully")
        return True

    def stop(self) -> None:
        """Stop the learner."""
        logger.info("Stopping learner...")
        self._running = False

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

    def run(self, total_timesteps: int = None) -> None:
        """Main training loop."""
        if not self._running:
            raise RuntimeError("Learner not started")

        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        # Collect rollouts until we have enough for a training batch
        rollouts_needed = self.num_workers
        collected_rollouts: List[RolloutBatch] = []

        while self._running and self.stats.total_timesteps < total_timesteps:
            try:
                # Collect rollouts from workers
                while len(collected_rollouts) < rollouts_needed:
                    rollout = self.rollout_queue.pop(timeout=self.config.learner_pop_timeout)

                    if rollout is None:
                        logger.warning("Timeout waiting for rollout")
                        continue

                    # Auto-detect obs_dim from first rollout
                    if self.model is None:
                        self.obs_dim = rollout.observations.shape[1]
                        logger.info(f"Auto-detected obs_dim={self.obs_dim} from first rollout")
                        self.model = self._create_model()
                        # Broadcast initial policy
                        self.stats.policy_version = 1
                        self._broadcast_policy()

                    collected_rollouts.append(rollout)
                    self.stats.log_rollout(rollout)
                    self.stats.total_timesteps += rollout.n_steps

                    logger.debug(
                        f"Received rollout from worker {rollout.worker_id}, "
                        f"collected {len(collected_rollouts)}/{rollouts_needed}"
                    )

                # Populate SB3's rollout buffer
                logger.info(f"Training update {self.stats.total_updates + 1}...")
                train_start = time.time()

                self._populate_rollout_buffer(collected_rollouts)

                # Use SB3's train() method
                self.model.train()

                train_time = time.time() - train_start

                # Update stats
                self.stats.total_updates += 1
                self.stats.policy_version += 1

                # Broadcast updated policy
                self._broadcast_policy()

                # Log to wandb
                wandb.log({
                    "train/timesteps": self.stats.total_timesteps,
                    "train/updates": self.stats.total_updates,
                    "train/train_time": train_time,
                    "train/policy_loss": self.model.logger.name_to_value.get("train/policy_gradient_loss", 0),
                    "train/value_loss": self.model.logger.name_to_value.get("train/value_loss", 0),
                    "train/entropy_loss": self.model.logger.name_to_value.get("train/entropy_loss", 0),
                    "train/approx_kl": self.model.logger.name_to_value.get("train/approx_kl", 0),
                    "train/clip_fraction": self.model.logger.name_to_value.get("train/clip_fraction", 0),
                    "system/queue_length": self.rollout_queue.length(),
                    "system/policy_version": self.stats.policy_version,
                })

                logger.info(
                    f"Update {self.stats.total_updates}: "
                    f"timesteps={self.stats.total_timesteps}, "
                    f"train_time={train_time:.1f}s"
                )

                # Reset for next batch
                collected_rollouts = []

                # Checkpoint
                if self.stats.total_updates % max(1, self.config.model_save_freq // (self.config.n_steps * self.num_workers)) == 0:
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

    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total timesteps")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-password", type=str, default=None, help="Redis password")
    parser.add_argument("--obs-dim", type=int, default=None, help="Observation dimension (auto-detect if not set)")
    parser.add_argument("--n-actions", type=int, default=9, help="Number of discrete actions")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to models dir (auto-finds latest)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [Learner] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=args.redis_password,
        total_timesteps=args.total_timesteps,
        checkpoint_dir=args.checkpoint_dir,
    )

    learner = Learner(
        config=config,
        num_workers=args.num_workers,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
    )

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        learner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if learner.start():
        # Load checkpoint if specified
        checkpoint_path = None
        if args.resume:
            checkpoint_path = Path(args.resume)
        elif args.models_dir:
            checkpoint_path = find_latest_model(Path(args.models_dir))

        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            custom_objects = {
                "clip_range": lambda _: 0.2,
                "lr_schedule": lambda _: config.learning_rate,
            }
            learner.model = PPO.load(checkpoint_path, device=learner.device, custom_objects=custom_objects)
            learner.obs_dim = learner.model.observation_space.shape[0]
            learner._broadcast_policy()
            logger.info("Checkpoint loaded and broadcast to workers")

        learner.run()
    else:
        logger.error("Failed to start learner")
        sys.exit(1)


if __name__ == "__main__":
    main()

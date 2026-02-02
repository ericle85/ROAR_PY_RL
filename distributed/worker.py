"""
Distributed PPO Worker (Python 3.8)

Collects rollouts from CARLA and sends to learner via Redis.
Each worker manages its own CARLA server instance.

Usage:
    python worker.py --worker-id 0 --carla-exe "C:/path/to/CarlaUE4.exe" --racing-line "path/to/racing_line.npz"
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
import nest_asyncio
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed.config import Config
from distributed.protocol import RolloutBatch, PolicyWeights
from distributed.redis_client import RolloutQueue, PolicyChannel, RedisHealthCheck
from distributed.carla_manager import CarlaManager

# For ROAR imports
from training.env_util import initialize_roar_env

nest_asyncio.apply()

logger = logging.getLogger(__name__)


class RolloutCollector:
    """
    Collects rollouts from the environment using a local policy copy.

    Similar to SB3's RolloutBuffer but designed for distributed collection.
    """

    def __init__(
        self,
        env: gym.Env,
        model: PPO,
        n_steps: int,
        worker_id: int,
    ):
        """
        Initialize rollout collector.

        Args:
            env: Gymnasium environment
            model: PPO model for local inference
            n_steps: Number of steps per rollout
            worker_id: Worker identifier
        """
        self.env = env
        self.model = model
        self.n_steps = n_steps
        self.worker_id = worker_id

        # Get observation and action dimensions
        obs_shape = env.observation_space.shape
        self.obs_dim = np.prod(obs_shape)

        # Pre-allocate buffers
        self.observations = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=bool)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.episode_starts = np.zeros(n_steps, dtype=bool)

        # Current state
        self._last_obs: Optional[np.ndarray] = None
        self._last_episode_start: bool = True

        # Statistics
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self._current_episode_reward: float = 0.0
        self._current_episode_length: int = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        obs, info = self.env.reset()
        self._last_obs = obs.flatten()
        self._last_episode_start = True
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        return self._last_obs

    def collect_rollout(self) -> RolloutBatch:
        """
        Collect a single rollout of n_steps.

        Returns:
            RolloutBatch with collected data
        """
        if self._last_obs is None:
            self.reset()

        self.model.policy.set_training_mode(False)

        for step in range(self.n_steps):
            # Store current observation
            self.observations[step] = self._last_obs
            self.episode_starts[step] = self._last_episode_start

            # Get action from policy
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).unsqueeze(0).to(self.model.device)
                action, value, log_prob = self.model.policy(obs_tensor)

            action_np = action.cpu().numpy().flatten()[0]
            value_np = value.cpu().numpy().flatten()[0]
            log_prob_np = log_prob.cpu().numpy().flatten()[0]

            # Store action, value, log_prob
            self.actions[step] = action_np
            self.values[step] = value_np
            self.log_probs[step] = log_prob_np

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.rewards[step] = reward
            self.dones[step] = done

            # Update episode stats
            self._current_episode_reward += reward
            self._current_episode_length += 1

            if done:
                # Log episode stats
                self.episode_rewards.append(self._current_episode_reward)
                self.episode_lengths.append(self._current_episode_length)
                logger.info(
                    f"Worker {self.worker_id} Episode done: "
                    f"reward={self._current_episode_reward:.2f}, "
                    f"length={self._current_episode_length}"
                )

                # Reset
                obs, info = self.env.reset()
                self._current_episode_reward = 0.0
                self._current_episode_length = 0
                self._last_episode_start = True
            else:
                self._last_episode_start = False

            self._last_obs = obs.flatten()

        # Compute value for last observation (needed for GAE)
        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).unsqueeze(0).to(self.model.device)
            _, last_value, _ = self.model.policy(obs_tensor)
            last_value_np = last_value.cpu().numpy().flatten()

        # Create rollout batch
        return RolloutBatch(
            worker_id=self.worker_id,
            observations=self.observations.copy(),
            actions=self.actions.copy(),
            rewards=self.rewards.copy(),
            dones=self.dones.copy(),
            values=self.values.copy(),
            log_probs=self.log_probs.copy(),
            episode_starts=self.episode_starts.copy(),
            last_values=last_value_np,
            last_dones=np.array([self._last_episode_start]),
        )

    def get_stats(self) -> Dict[str, float]:
        """Get collection statistics."""
        stats = {}
        if self.episode_rewards:
            stats["mean_episode_reward"] = np.mean(self.episode_rewards[-100:])
            stats["mean_episode_length"] = np.mean(self.episode_lengths[-100:])
            stats["num_episodes"] = len(self.episode_rewards)
        return stats


class Worker:
    """
    Distributed worker that collects rollouts from CARLA.

    Handles:
    - CARLA server lifecycle
    - Environment initialization
    - Rollout collection
    - Policy weight updates from learner
    """

    def __init__(
        self,
        worker_id: int,
        config: Config,
        carla_exe_path: str,
        racing_line_path: str,
    ):
        """
        Initialize worker.

        Args:
            worker_id: Unique worker identifier (0, 1, 2, ...)
            config: Configuration object
            carla_exe_path: Path to CARLA executable
            racing_line_path: Path to racing line NPZ file
        """
        self.worker_id = worker_id
        self.config = config
        self.carla_exe_path = carla_exe_path
        self.racing_line_path = racing_line_path

        # Calculate port for this worker
        self.carla_port = config.get_carla_port(worker_id)

        # Components (initialized in start())
        self.carla_manager: Optional[CarlaManager] = None
        self.env: Optional[gym.Env] = None
        self.model: Optional[PPO] = None
        self.collector: Optional[RolloutCollector] = None
        self.rollout_queue: Optional[RolloutQueue] = None
        self.policy_channel: Optional[PolicyChannel] = None

        # State
        self.policy_version: int = 0
        self.total_steps: int = 0
        self.total_rollouts: int = 0
        self._running: bool = False

    def _initialize_env(self) -> gym.Env:
        """Initialize ROAR environment."""
        logger.info(f"Initializing environment on port {self.carla_port}...")

        env = asyncio.get_event_loop().run_until_complete(
            initialize_roar_env(
                carla_host=self.config.carla_host,
                carla_port=self.carla_port,
                control_timestep=self.config.control_timestep,
                physics_timestep=self.config.physics_timestep,
                racing_line_path=self.racing_line_path,
                use_discrete_actions=self.config.use_discrete_actions,
            )
        )

        # Apply standard wrappers
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=self.config.time_limit_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        logger.info(f"Environment initialized. Obs shape: {env.observation_space.shape}, "
                    f"Action space: {env.action_space}")

        return env

    def _initialize_model(self) -> PPO:
        """Initialize PPO model for local inference."""
        logger.info("Initializing PPO model...")

        # Create model with same architecture as learner
        # Note: We don't need full training capabilities, just inference
        model = PPO(
            "MlpPolicy",
            self.env,
            **self.config.get_ppo_params(),
            device="cpu",  # Workers use CPU
        )

        return model

    def _update_policy(self, weights: PolicyWeights) -> bool:
        """
        Update local policy with weights from learner.

        Args:
            weights: PolicyWeights from learner

        Returns:
            bool: True if update successful
        """
        try:
            # Convert numpy arrays back to torch tensors
            state_dict = {}
            for key, value in weights.state_dict.items():
                state_dict[key] = th.from_numpy(value)

            # Load into model
            self.model.policy.load_state_dict(state_dict)
            self.policy_version = weights.version

            logger.info(f"Updated policy to version {self.policy_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to update policy: {e}")
            return False

    def start(self) -> bool:
        """
        Start the worker.

        Initializes all components:
        - Redis connection
        - CARLA server
        - Environment
        - Model

        Returns:
            bool: True if started successfully
        """
        logger.info(f"Starting worker {self.worker_id}...")

        # Check Redis connection
        health = RedisHealthCheck(self.config)
        if not health.wait_for_redis(timeout=30.0):
            logger.error("Redis not available")
            return False

        # Initialize Redis clients
        self.rollout_queue = RolloutQueue(self.config)
        self.policy_channel = PolicyChannel(self.config)

        # Start subscription to policy updates
        self.policy_channel.subscribe()

        # Start CARLA server
        self.carla_manager = CarlaManager(
            carla_exe_path=self.carla_exe_path,
            port=self.carla_port,
            config=self.config,
        )

        if not self.carla_manager.start():
            logger.error("Failed to start CARLA server")
            return False

        # Initialize environment
        try:
            self.env = self._initialize_env()
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            self.carla_manager.stop()
            return False

        # Initialize model
        self.model = self._initialize_model()

        # Try to get initial policy from learner
        initial_weights = self.policy_channel.get_latest()
        if initial_weights is not None:
            self._update_policy(initial_weights)
            logger.info(f"Loaded initial policy version {self.policy_version}")
        else:
            logger.info("No initial policy available, using random initialization")

        # Initialize collector
        self.collector = RolloutCollector(
            env=self.env,
            model=self.model,
            n_steps=self.config.n_steps,
            worker_id=self.worker_id,
        )

        self._running = True
        logger.info(f"Worker {self.worker_id} started successfully")
        return True

    def stop(self) -> None:
        """Stop the worker and clean up resources."""
        logger.info(f"Stopping worker {self.worker_id}...")
        self._running = False

        if self.policy_channel is not None:
            self.policy_channel.unsubscribe()

        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")

        if self.carla_manager is not None:
            self.carla_manager.stop()

        logger.info(f"Worker {self.worker_id} stopped")

    def run(self) -> None:
        """
        Main worker loop.

        Continuously:
        1. Check for policy updates
        2. Collect rollouts
        3. Send rollouts to learner
        """
        if not self._running:
            raise RuntimeError("Worker not started")

        logger.info(f"Worker {self.worker_id} entering main loop...")

        while self._running:
            try:
                # Check for policy updates (non-blocking)
                update = self.policy_channel.check_for_update(self.policy_version)
                if update is not None:
                    self._update_policy(update)

                # Collect rollout
                logger.debug(f"Collecting rollout {self.total_rollouts}...")
                start_time = time.time()

                rollout = self.collector.collect_rollout()

                collection_time = time.time() - start_time
                self.total_steps += rollout.n_steps
                self.total_rollouts += 1

                # Send to learner
                queue_len = self.rollout_queue.push(rollout)

                # Log progress
                stats = self.collector.get_stats()
                logger.info(
                    f"Worker {self.worker_id} Rollout {self.total_rollouts}: "
                    f"steps={self.total_steps}, time={collection_time:.1f}s, "
                    f"queue_len={queue_len}, policy_v={self.policy_version}"
                )
                if stats:
                    logger.info(
                        f"  Episodes: {stats.get('num_episodes', 0)}, "
                        f"Mean reward: {stats.get('mean_episode_reward', 0):.2f}"
                    )

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break

            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)

                # Attempt recovery
                if not self._attempt_recovery():
                    logger.error("Recovery failed, stopping worker")
                    break

        self.stop()

    def _attempt_recovery(self) -> bool:
        """
        Attempt to recover from errors.

        Tries to restart CARLA and reinitialize environment.

        Returns:
            bool: True if recovery successful
        """
        logger.info("Attempting recovery...")

        try:
            # Close old environment
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = None

            # Restart CARLA
            if not self.carla_manager.restart():
                logger.error("Failed to restart CARLA")
                return False

            # Reinitialize environment
            self.env = self._initialize_env()

            # Reinitialize collector with new env
            self.collector = RolloutCollector(
                env=self.env,
                model=self.model,
                n_steps=self.config.n_steps,
                worker_id=self.worker_id,
            )

            logger.info("Recovery successful")
            return True

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False


def main():
    """Main entry point for worker."""
    parser = argparse.ArgumentParser(description="Distributed PPO Worker")

    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Worker ID (0, 1, 2, ...)",
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
        "--carla-exe",
        type=str,
        required=True,
        help="Path to CarlaUE4.exe",
    )
    parser.add_argument(
        "--racing-line",
        type=str,
        required=True,
        help="Path to racing line .npz file",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per rollout",
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
        format=f"%(asctime)s [Worker {args.worker_id}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create config
    config = Config(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=args.redis_password,
        racing_line_path=args.racing_line,
        n_steps=args.n_steps,
    )

    # Create and run worker
    worker = Worker(
        worker_id=args.worker_id,
        config=config,
        carla_exe_path=args.carla_exe,
        racing_line_path=args.racing_line,
    )

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start and run
    if worker.start():
        worker.run()
    else:
        logger.error("Failed to start worker")
        sys.exit(1)


if __name__ == "__main__":
    main()

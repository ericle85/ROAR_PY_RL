"""
Shared configuration for distributed PPO training.

This module contains all configuration constants and hyperparameters
used by both workers and the learner.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration for distributed PPO training."""

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None  # Set if Redis requires authentication
    rollout_queue_key: str = "rollouts"
    policy_channel: str = "policy_updates"

    # CARLA configuration
    carla_host: str = "localhost"
    carla_base_port: int = 2000
    carla_port_stride: int = 10
    carla_exe_path: str = ""  # Must be set by user
    carla_startup_wait: float = 10.0  # Seconds to wait for CARLA to start
    carla_connection_timeout: float = 15.0

    # Environment configuration
    control_timestep: float = 1.0 / 25  # 25 FPS
    physics_timestep: float = 1.0 / 125  # 125 Hz physics (5 substeps)
    time_limit_seconds: int = 120  # 2 minutes per episode
    racing_line_path: str = ""  # Path to racing line .npz file
    use_discrete_actions: bool = True

    # PPO hyperparameters (from train_online.py)
    learning_rate: float = 2.5e-4
    n_steps: int = 2048  # Rollout buffer size per worker
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01  # Entropy coefficient for exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # Training configuration
    total_timesteps: int = 10_000_000
    model_save_freq: int = 50_000
    checkpoint_dir: str = "checkpoints"

    # Worker configuration
    worker_rollout_timeout: float = 300.0  # Max time for rollout collection
    worker_restart_delay: float = 5.0  # Delay before restarting failed CARLA

    # Learner configuration
    learner_pop_timeout: float = 60.0  # Timeout for blocking pop

    # Logging
    wandb_project: str = "ROAR_PY_RL_Distributed"
    log_interval: int = 1  # Log every N updates

    def get_carla_port(self, worker_id: int) -> int:
        """Calculate CARLA port for a given worker ID."""
        return self.carla_base_port + worker_id * self.carla_port_stride

    def get_ppo_params(self) -> dict:
        """Get PPO hyperparameters as a dictionary for stable-baselines3."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "verbose": 1,
        }

    @property
    def time_limit_steps(self) -> int:
        """Get time limit in environment steps."""
        return int(self.time_limit_seconds / self.control_timestep)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.carla_exe_path and not os.path.exists(self.carla_exe_path):
            errors.append(f"CARLA executable not found: {self.carla_exe_path}")

        if self.racing_line_path and not os.path.exists(self.racing_line_path):
            errors.append(f"Racing line file not found: {self.racing_line_path}")

        if self.n_steps < self.batch_size:
            errors.append(f"n_steps ({self.n_steps}) must be >= batch_size ({self.batch_size})")

        return errors


# Default configuration instance
default_config = Config()

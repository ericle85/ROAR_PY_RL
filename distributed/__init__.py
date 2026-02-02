"""
Distributed PPO training module for ROAR_PY_RL.

This module implements a multi-worker architecture where:
- Workers (Python 3.8): Collect rollouts from CARLA servers
- Learner (Python 3.11): Aggregates rollouts and trains PPO on GPU

Architecture:
    - Redis server for IPC (rollout queue + policy pub/sub)
    - Workers push rollouts to Redis queue
    - Learner pops rollouts, trains PPO, broadcasts weights

Usage:
    1. Start Redis server
    2. Start learner: python distributed/learner.py --num-workers 2
    3. Start workers: python distributed/worker.py --worker-id 0 ...
                      python distributed/worker.py --worker-id 1 ...
"""

from .config import Config
from .protocol import RolloutBatch, PolicyWeights
from .redis_client import RolloutQueue, PolicyChannel, RedisHealthCheck
from .carla_manager import CarlaManager

__all__ = [
    'Config',
    'RolloutBatch',
    'PolicyWeights',
    'RolloutQueue',
    'PolicyChannel',
    'RedisHealthCheck',
    'CarlaManager',
]

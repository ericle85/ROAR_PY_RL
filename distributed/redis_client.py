"""
Redis client helpers for distributed PPO training.

Provides high-level interfaces for:
- RolloutQueue: Push/pop rollout batches
- PolicyChannel: Publish/subscribe policy weights
"""

import time
import threading
from typing import Optional, Callable
import redis

from .protocol import RolloutBatch, PolicyWeights
from .config import Config, default_config


class RolloutQueue:
    """
    Redis-backed queue for rollout batches.

    Workers push rollouts with LPUSH, learner pops with BRPOP (FIFO order).
    """

    def __init__(
        self,
        config: Config = None,
        redis_client: redis.Redis = None,
    ):
        """
        Initialize rollout queue.

        Args:
            config: Configuration object (uses default if None)
            redis_client: Existing Redis client (creates new if None)
        """
        self.config = config or default_config

        if redis_client is not None:
            self.redis = redis_client
        else:
            self.redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=False,  # Need binary for msgpack
            )

        self.queue_key = self.config.rollout_queue_key

    def push(self, rollout: RolloutBatch) -> int:
        """
        Push a rollout batch to the queue.

        Args:
            rollout: RolloutBatch to push

        Returns:
            int: Current queue length after push
        """
        serialized = rollout.serialize()
        return self.redis.lpush(self.queue_key, serialized)

    def pop(self, timeout: float = None) -> Optional[RolloutBatch]:
        """
        Pop a rollout batch from the queue (blocking).

        Args:
            timeout: Timeout in seconds (None = use config default)

        Returns:
            RolloutBatch if available, None if timeout
        """
        if timeout is None:
            timeout = self.config.learner_pop_timeout

        # BRPOP returns (key, value) tuple or None on timeout
        result = self.redis.brpop(self.queue_key, timeout=int(timeout))
        if result is None:
            return None

        _, data = result
        return RolloutBatch.deserialize(data)

    def pop_nonblocking(self) -> Optional[RolloutBatch]:
        """
        Pop a rollout batch without blocking.

        Returns:
            RolloutBatch if available, None if queue empty
        """
        data = self.redis.rpop(self.queue_key)
        if data is None:
            return None
        return RolloutBatch.deserialize(data)

    def length(self) -> int:
        """Get current queue length."""
        return self.redis.llen(self.queue_key)

    def clear(self) -> int:
        """Clear all rollouts from queue. Returns number deleted."""
        return self.redis.delete(self.queue_key)


class PolicyChannel:
    """
    Redis pub/sub channel for policy weight broadcasts.

    Learner publishes updated weights, workers subscribe to receive them.
    Also stores latest weights in a key for workers joining late.
    """

    def __init__(
        self,
        config: Config = None,
        redis_client: redis.Redis = None,
    ):
        """
        Initialize policy channel.

        Args:
            config: Configuration object (uses default if None)
            redis_client: Existing Redis client (creates new if None)
        """
        self.config = config or default_config

        if redis_client is not None:
            self.redis = redis_client
        else:
            self.redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=False,
            )

        self.channel_name = self.config.policy_channel
        self.latest_key = f"{self.channel_name}:latest"
        self.version_key = f"{self.channel_name}:version"

        # For subscription
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscriber_thread: Optional[threading.Thread] = None
        self._latest_weights: Optional[PolicyWeights] = None
        self._callback: Optional[Callable[[PolicyWeights], None]] = None
        self._running = False

    def publish(self, weights: PolicyWeights) -> int:
        """
        Publish policy weights to all subscribers.

        Also stores in a key so late-joining workers can get latest.

        Args:
            weights: PolicyWeights to broadcast

        Returns:
            int: Number of subscribers that received the message
        """
        serialized = weights.serialize()

        # Store latest for late joiners
        self.redis.set(self.latest_key, serialized)
        self.redis.set(self.version_key, weights.version)

        # Broadcast to subscribers
        return self.redis.publish(self.channel_name, serialized)

    def get_latest(self) -> Optional[PolicyWeights]:
        """
        Get the latest published policy weights.

        Useful for workers joining after learner has started.

        Returns:
            PolicyWeights if available, None if no weights published yet
        """
        data = self.redis.get(self.latest_key)
        if data is None:
            return None
        return PolicyWeights.deserialize(data)

    def get_version(self) -> int:
        """Get the current policy version number."""
        version = self.redis.get(self.version_key)
        if version is None:
            return 0
        return int(version)

    def subscribe(
        self,
        callback: Callable[[PolicyWeights], None] = None,
    ) -> None:
        """
        Subscribe to policy weight updates.

        Starts a background thread that listens for updates.
        When update received, either calls callback or stores in _latest_weights.

        Args:
            callback: Optional callback function called with new PolicyWeights
        """
        if self._running:
            return

        self._callback = callback
        self._pubsub = self.redis.pubsub()
        self._pubsub.subscribe(self.channel_name)

        self._running = True
        self._subscriber_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._subscriber_thread.start()

    def _listen_loop(self):
        """Background thread that listens for policy updates."""
        try:
            for message in self._pubsub.listen():
                if not self._running:
                    break

                if message["type"] != "message":
                    continue

                try:
                    weights = PolicyWeights.deserialize(message["data"])
                    self._latest_weights = weights

                    if self._callback is not None:
                        self._callback(weights)
                except Exception as e:
                    print(f"Error deserializing policy weights: {e}")
        except (OSError, ValueError, redis.ConnectionError):
            # Socket closed during unsubscribe - this is expected
            pass

    def get_update(self) -> Optional[PolicyWeights]:
        """
        Get latest weights received via subscription (non-blocking).

        Returns:
            PolicyWeights if new update received, None otherwise
        """
        weights = self._latest_weights
        self._latest_weights = None  # Clear after reading
        return weights

    def check_for_update(self, current_version: int) -> Optional[PolicyWeights]:
        """
        Check if there's a newer policy version available.

        Args:
            current_version: Worker's current policy version

        Returns:
            PolicyWeights if newer version available, None otherwise
        """
        # First check subscription for real-time update
        update = self.get_update()
        if update is not None and update.version > current_version:
            return update

        # Fall back to checking stored version
        stored_version = self.get_version()
        if stored_version > current_version:
            return self.get_latest()

        return None

    def unsubscribe(self) -> None:
        """Stop subscription and clean up."""
        self._running = False

        if self._pubsub is not None:
            self._pubsub.unsubscribe()
            self._pubsub.close()
            self._pubsub = None

        if self._subscriber_thread is not None:
            self._subscriber_thread.join(timeout=1.0)
            self._subscriber_thread = None

    def clear(self) -> None:
        """Clear stored policy weights."""
        self.redis.delete(self.latest_key)
        self.redis.delete(self.version_key)


class RedisHealthCheck:
    """Utility for checking Redis connection health."""

    def __init__(self, config: Config = None):
        self.config = config or default_config

    def check(self) -> bool:
        """
        Check if Redis is available and responding.

        Returns:
            bool: True if Redis is healthy
        """
        try:
            client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                socket_timeout=5.0,
            )
            return client.ping()
        except (redis.ConnectionError, redis.AuthenticationError):
            return False

    def wait_for_redis(self, timeout: float = 30.0) -> bool:
        """
        Wait for Redis to become available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if Redis became available, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.check():
                return True
            time.sleep(1.0)
        return False


def test_redis_client():
    """Test Redis client functionality."""
    config = Config()

    # Test health check
    health = RedisHealthCheck(config)
    if not health.check():
        print("Redis not available, skipping test")
        return

    print("Redis is healthy")

    # Test rollout queue
    queue = RolloutQueue(config)
    queue.clear()

    import numpy as np

    rollout = RolloutBatch(
        worker_id=0,
        observations=np.random.randn(10, 64).astype(np.float32),
        actions=np.random.randint(0, 9, size=(10,)),
        rewards=np.random.randn(10).astype(np.float32),
        dones=np.zeros(10, dtype=bool),
        values=np.random.randn(10).astype(np.float32),
        log_probs=np.random.randn(10).astype(np.float32),
        episode_starts=np.zeros(10, dtype=bool),
        last_values=np.array([0.0], dtype=np.float32),
        last_dones=np.array([False]),
    )

    # Push and pop
    queue.push(rollout)
    print(f"Queue length: {queue.length()}")

    popped = queue.pop(timeout=1)
    assert popped is not None
    assert popped.worker_id == 0
    print("Rollout queue test passed!")

    # Test policy channel
    channel = PolicyChannel(config)
    channel.clear()

    weights = PolicyWeights(
        version=1,
        state_dict={"layer.weight": np.random.randn(10, 10).astype(np.float32)},
    )

    channel.publish(weights)
    print(f"Policy version: {channel.get_version()}")

    latest = channel.get_latest()
    assert latest is not None
    assert latest.version == 1
    print("Policy channel test passed!")

    # Test subscription
    received = []

    def on_weights(w):
        received.append(w)

    channel.subscribe(callback=on_weights)
    time.sleep(0.5)  # Let subscriber thread start

    weights2 = PolicyWeights(
        version=2,
        state_dict={"layer.weight": np.random.randn(10, 10).astype(np.float32)},
    )
    channel.publish(weights2)

    time.sleep(0.5)  # Wait for message
    channel.unsubscribe()

    assert len(received) == 1
    assert received[0].version == 2
    print("Subscription test passed!")

    print("All Redis client tests passed!")


if __name__ == "__main__":
    test_redis_client()

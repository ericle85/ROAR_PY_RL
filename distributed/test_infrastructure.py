#!/usr/bin/env python
"""
Test script for distributed training infrastructure.

This script verifies that all components work correctly:
- Protocol serialization/deserialization
- Redis client operations
- Configuration

Run this before starting actual training to ensure everything is set up correctly.

Usage:
    python distributed/test_infrastructure.py [--redis-host localhost] [--redis-port 6379]
"""

import argparse
import os
import sys
import time

# Add repo root to path for imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Verify we can find the distributed module
if not os.path.exists(os.path.join(REPO_ROOT, "distributed", "__init__.py")):
    print(f"ERROR: Cannot find distributed module at {REPO_ROOT}")
    print("Make sure you're running from the ROAR_PY_RL directory")
    sys.exit(1)

import numpy as np

# Test results tracking
tests_passed = 0
tests_failed = 0


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global tests_passed, tests_failed
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                func(*args, **kwargs)
                print(f"✓ PASSED: {name}")
                tests_passed += 1
            except Exception as e:
                print(f"✗ FAILED: {name}")
                print(f"  Error: {e}")
                tests_failed += 1
        return wrapper
    return decorator


@test("Protocol - RolloutBatch serialization")
def test_rollout_serialization():
    from distributed.protocol import RolloutBatch

    # Create test data
    n_steps = 100
    obs_dim = 94

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
    rollout.dones[50] = True
    rollout.episode_starts[51] = True

    # Serialize
    serialized = rollout.serialize()
    print(f"  Serialized size: {len(serialized):,} bytes")

    # Deserialize
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
    assert np.allclose(deserialized.last_values, rollout.last_values)
    assert np.array_equal(deserialized.last_dones, rollout.last_dones)


@test("Protocol - PolicyWeights serialization")
def test_policy_weights_serialization():
    from distributed.protocol import PolicyWeights

    weights = PolicyWeights(
        version=42,
        state_dict={
            "policy.mlp_extractor.policy_net.0.weight": np.random.randn(64, 94).astype(np.float32),
            "policy.mlp_extractor.policy_net.0.bias": np.random.randn(64).astype(np.float32),
            "policy.mlp_extractor.policy_net.2.weight": np.random.randn(64, 64).astype(np.float32),
            "policy.mlp_extractor.policy_net.2.bias": np.random.randn(64).astype(np.float32),
            "policy.action_net.weight": np.random.randn(9, 64).astype(np.float32),
            "policy.action_net.bias": np.random.randn(9).astype(np.float32),
        },
    )

    # Serialize
    serialized = weights.serialize()
    print(f"  Serialized size: {len(serialized):,} bytes")

    # Deserialize
    deserialized = PolicyWeights.deserialize(serialized)

    # Verify
    assert deserialized.version == weights.version
    for key in weights.state_dict:
        assert key in deserialized.state_dict
        assert np.allclose(deserialized.state_dict[key], weights.state_dict[key])


@test("Config - PPO parameters")
def test_config():
    from distributed.config import Config

    config = Config()

    # Check port calculation
    assert config.get_carla_port(0) == 2000
    assert config.get_carla_port(1) == 2010
    assert config.get_carla_port(2) == 2020

    # Check PPO params
    ppo_params = config.get_ppo_params()
    assert "learning_rate" in ppo_params
    assert ppo_params["n_steps"] == 2048
    assert ppo_params["batch_size"] == 64
    assert ppo_params["gamma"] == 0.99

    print(f"  CARLA ports: {[config.get_carla_port(i) for i in range(3)]}")
    print(f"  PPO params: learning_rate={ppo_params['learning_rate']}, n_steps={ppo_params['n_steps']}")


@test("Redis - Connection")
def test_redis_connection(redis_host, redis_port, redis_password):
    from distributed.config import Config
    from distributed.redis_client import RedisHealthCheck

    config = Config(redis_host=redis_host, redis_port=redis_port, redis_password=redis_password)
    health = RedisHealthCheck(config)

    if not health.check():
        raise RuntimeError(
            f"Redis not available at {redis_host}:{redis_port}\n"
            "  Please start Redis server and try again."
        )
    print(f"  Connected to Redis at {redis_host}:{redis_port}")


@test("Redis - RolloutQueue operations")
def test_rollout_queue(redis_host, redis_port, redis_password):
    from distributed.config import Config
    from distributed.protocol import RolloutBatch
    from distributed.redis_client import RolloutQueue

    config = Config(redis_host=redis_host, redis_port=redis_port, redis_password=redis_password)
    queue = RolloutQueue(config)

    # Clear any existing data
    queue.clear()
    assert queue.length() == 0

    # Create test rollout
    rollout = RolloutBatch(
        worker_id=0,
        observations=np.random.randn(10, 94).astype(np.float32),
        actions=np.random.randint(0, 9, size=(10,)),
        rewards=np.random.randn(10).astype(np.float32),
        dones=np.zeros(10, dtype=bool),
        values=np.random.randn(10).astype(np.float32),
        log_probs=np.random.randn(10).astype(np.float32),
        episode_starts=np.zeros(10, dtype=bool),
        last_values=np.array([0.0], dtype=np.float32),
        last_dones=np.array([False]),
    )

    # Push
    queue.push(rollout)
    assert queue.length() == 1
    print("  Push: OK")

    # Push another
    rollout.worker_id = 1
    queue.push(rollout)
    assert queue.length() == 2
    print("  Multiple pushes: OK")

    # Pop (FIFO order)
    popped = queue.pop(timeout=1)
    assert popped is not None
    assert popped.worker_id == 0  # First in, first out
    print("  Pop (FIFO): OK")

    # Non-blocking pop
    popped = queue.pop_nonblocking()
    assert popped is not None
    assert popped.worker_id == 1
    print("  Non-blocking pop: OK")

    # Queue should be empty
    assert queue.length() == 0
    popped = queue.pop_nonblocking()
    assert popped is None
    print("  Empty queue handling: OK")

    # Clean up
    queue.clear()


@test("Redis - PolicyChannel operations")
def test_policy_channel(redis_host, redis_port, redis_password):
    from distributed.config import Config
    from distributed.protocol import PolicyWeights
    from distributed.redis_client import PolicyChannel

    config = Config(redis_host=redis_host, redis_port=redis_port, redis_password=redis_password)
    channel = PolicyChannel(config)

    # Clear any existing data
    channel.clear()
    assert channel.get_version() == 0
    assert channel.get_latest() is None

    # Create test weights
    weights = PolicyWeights(
        version=1,
        state_dict={"test.weight": np.random.randn(10, 10).astype(np.float32)},
    )

    # Publish
    channel.publish(weights)
    print("  Publish: OK")

    # Check version
    assert channel.get_version() == 1
    print("  Version tracking: OK")

    # Get latest
    latest = channel.get_latest()
    assert latest is not None
    assert latest.version == 1
    print("  Get latest: OK")

    # Test subscription
    received = []

    def on_weights(w):
        received.append(w)

    channel.subscribe(callback=on_weights)
    time.sleep(0.5)  # Let subscriber start

    weights2 = PolicyWeights(
        version=2,
        state_dict={"test.weight": np.random.randn(10, 10).astype(np.float32)},
    )
    channel.publish(weights2)
    time.sleep(0.5)  # Wait for message

    channel.unsubscribe()

    assert len(received) == 1
    assert received[0].version == 2
    print("  Pub/sub: OK")

    # Check for update
    channel.subscribe()
    time.sleep(0.2)

    update = channel.check_for_update(current_version=1)
    assert update is not None
    assert update.version == 2
    print("  Check for update: OK")

    channel.unsubscribe()
    channel.clear()


@test("Integration - Multiple workers simulation")
def test_multi_worker_simulation(redis_host, redis_port, redis_password):
    """Simulate multiple workers sending rollouts."""
    from distributed.config import Config
    from distributed.protocol import RolloutBatch, PolicyWeights
    from distributed.redis_client import RolloutQueue, PolicyChannel

    config = Config(redis_host=redis_host, redis_port=redis_port, redis_password=redis_password)
    queue = RolloutQueue(config)
    channel = PolicyChannel(config)

    queue.clear()
    channel.clear()

    # Simulate learner publishing initial policy
    initial_weights = PolicyWeights(
        version=1,
        state_dict={"layer.weight": np.random.randn(10, 10).astype(np.float32)},
    )
    channel.publish(initial_weights)

    # Simulate 3 workers sending rollouts
    num_workers = 3
    n_steps = 100

    for worker_id in range(num_workers):
        rollout = RolloutBatch(
            worker_id=worker_id,
            observations=np.random.randn(n_steps, 94).astype(np.float32),
            actions=np.random.randint(0, 9, size=(n_steps,)),
            rewards=np.random.randn(n_steps).astype(np.float32),
            dones=np.zeros(n_steps, dtype=bool),
            values=np.random.randn(n_steps).astype(np.float32),
            log_probs=np.random.randn(n_steps).astype(np.float32),
            episode_starts=np.zeros(n_steps, dtype=bool),
            last_values=np.array([0.0], dtype=np.float32),
            last_dones=np.array([False]),
        )
        queue.push(rollout)

    assert queue.length() == num_workers
    print(f"  {num_workers} workers sent rollouts: OK")

    # Simulate learner receiving rollouts
    received_workers = set()
    for _ in range(num_workers):
        rollout = queue.pop(timeout=1)
        assert rollout is not None
        received_workers.add(rollout.worker_id)

    assert received_workers == {0, 1, 2}
    print("  Learner received all rollouts: OK")

    # Simulate learner broadcasting updated policy
    updated_weights = PolicyWeights(
        version=2,
        state_dict={"layer.weight": np.random.randn(10, 10).astype(np.float32)},
    )
    channel.publish(updated_weights)

    # Workers can get update
    latest = channel.get_latest()
    assert latest.version == 2
    print("  Policy broadcast received: OK")

    # Clean up
    queue.clear()
    channel.clear()


def main():
    parser = argparse.ArgumentParser(description="Test distributed training infrastructure")
    parser.add_argument("--redis-host", default="localhost", help="Redis server host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port")
    parser.add_argument("--redis-password", default=None, help="Redis server password")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DISTRIBUTED TRAINING INFRASTRUCTURE TESTS")
    print("="*60)

    # Run tests that don't need Redis first
    test_rollout_serialization()
    test_policy_weights_serialization()
    test_config()

    # Redis tests
    try:
        test_redis_connection(args.redis_host, args.redis_port, args.redis_password)
        test_rollout_queue(args.redis_host, args.redis_port, args.redis_password)
        test_policy_channel(args.redis_host, args.redis_port, args.redis_password)
        test_multi_worker_simulation(args.redis_host, args.redis_port, args.redis_password)
    except Exception as e:
        print(f"\n⚠ Redis tests skipped: {e}")
        print("  Start Redis server to run all tests.")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  Passed: {tests_passed}")
    print(f"  Failed: {tests_failed}")
    print("="*60)

    if tests_failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed! Infrastructure is ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()

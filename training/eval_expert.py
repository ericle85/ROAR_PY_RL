"""
Evaluate the #1 hardcoded expert policy in the ROAR environment.

This script runs their RoarCompetitionSolution to verify it works correctly
before using it for imitation learning / behavioral cloning.

Note: Their code is optimized to start from the competition start location (waypoint 0).
The environment spawn point should match their expected start position.
"""

import gymnasium as gym
import numpy as np
import asyncio
import nest_asyncio
from pathlib import Path
import argparse

from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper
from expert_policy import ExpertPolicyWrapper, ExpertRunner


RUN_FPS = 25
SUBSTEPS_PER_STEP = 5
RACING_LINE_PATH = r"C:\Users\shrek\ROAR_PY_RL\ROAR_1_S25\competition_code\waypoints\waypointsPrimary.npz"
CENTERLINE_PATH = r"C:\Users\shrek\ROAR_PY_RL\racingline\Monza.npz"


async def get_env_async(record_video: bool = False, video_name: str = "expert_eval", spawn_at_start: bool = True) -> gym.Env:
    """Initialize the ROAR environment."""
    env = await initialize_roar_env(
        control_timestep=1.0 / RUN_FPS,
        physics_timestep=1.0 / (RUN_FPS * SUBSTEPS_PER_STEP),
        image_width=1920,
        image_height=1080,
        racing_line_path=str(RACING_LINE_PATH),
        centerline_path=str(CENTERLINE_PATH),
        use_discrete_actions=False  # Continuous actions for expert
    )

    # Set spawn location - waypoint 0 for competition start
    if spawn_at_start:
        env.unwrapped.spawn_waypoint_idx = 0

    # Keep unwrapped env reference before adding wrappers
    unwrapped_env = env

    # Add wrappers
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{video_name}",
            episode_trigger=lambda x: True  # Record all episodes
        )

    return env


async def run_expert_async(
    max_steps: int = 3000,
    episodes: int = 1,
    record_video: bool = False,
    collect_data: bool = False,
    verbose: bool = True
):
    """Run the expert policy."""
    print("Initializing environment...")
    env = await get_env_async(record_video=record_video)

    print("Creating expert policy wrapper...")
    expert = ExpertPolicyWrapper(env)

    # Note: Expert will be initialized by runner AFTER env.reset()
    # to ensure waypoints are computed from the correct spawn location
    runner = ExpertRunner(env, expert)
    results = []

    try:
        for ep in range(episodes):
            print(f"\n=== Episode {ep + 1}/{episodes} ===")
            result = await runner.run_episode(
                max_steps=max_steps,
                collect_data=collect_data,
                verbose=verbose
            )
            results.append(result)
            print(f"Episode result: reward={result['total_reward']:.1f}, steps={result['steps']}")

            if collect_data:
                # Save collected data
                save_path = Path(f"expert_data/episode_{ep}.npz")
                save_path.parent.mkdir(exist_ok=True)
                np.savez(
                    save_path,
                    observations=result['observations'],
                    actions=result['actions']
                )
                print(f"Saved {len(result['observations'])} transitions to {save_path}")

    finally:
        env.close()

    # Summary
    if len(results) > 1:
        print("\n=== Summary ===")
        rewards = [r['total_reward'] for r in results]
        steps = [r['steps'] for r in results]
        print(f"Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Average steps: {np.mean(steps):.0f} +/- {np.std(steps):.0f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate #1 hardcoded expert policy")
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=3000,
                        help='Max steps per episode')
    parser.add_argument('--record', action='store_true',
                        help='Record video of episodes')
    parser.add_argument('--collect', action='store_true',
                        help='Collect (observation, action) pairs for imitation learning')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose output')
    args = parser.parse_args()

    nest_asyncio.apply()

    asyncio.run(run_expert_async(
        max_steps=args.max_steps,
        episodes=args.episodes,
        record_video=args.record,
        collect_data=args.collect,
        verbose=not args.quiet
    ))


if __name__ == "__main__":
    main()

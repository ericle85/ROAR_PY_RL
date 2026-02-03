"""
Evaluate pretrained behavioral cloning model in the ROAR environment.

Spawns at the competition start location (waypoint 0) to match expert training data.
"""

import gymnasium as gym
import numpy as np
import asyncio
import nest_asyncio
from pathlib import Path
import argparse
import torch
import tqdm

from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper
from supervised.model import MLPPolicy


RUN_FPS = 25
SUBSTEPS_PER_STEP = 5
RACING_LINE_PATH = Path(__file__).parent.parent / "racingline" / "main.npz"


def load_model(checkpoint_path: str, device: torch.device) -> MLPPolicy:
    """Load pretrained BC model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = MLPPolicy(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_sizes=config["hidden_sizes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Config: {config}")
    if "val_loss" in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    if "val_acc" in checkpoint:
        print(f"  Val acc: {checkpoint['val_acc']:.4f}")

    return model


async def get_env_async(
    record_video: bool = False,
    video_name: str = "pretrained_eval",
    spawn_at_start: bool = True
) -> gym.Env:
    """Initialize the ROAR environment."""
    env = await initialize_roar_env(
        control_timestep=1.0 / RUN_FPS,
        physics_timestep=1.0 / (RUN_FPS * SUBSTEPS_PER_STEP),
        image_width=1920,
        image_height=1080,
        racing_line_path=str(RACING_LINE_PATH),
        use_discrete_actions=False,
    )

    # Set spawn location - waypoint 0 for competition start
    if spawn_at_start:
        env.unwrapped.spawn_waypoint_idx = 0

    # Add wrappers
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{video_name}",
            episode_trigger=lambda x: True,
        )

    return env


async def run_pretrained_async(
    checkpoint_path: str,
    max_steps: int = 3000,
    episodes: int = 1,
    record_video: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
):
    """Run the pretrained BC model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Initialize environment
    print("Initializing environment...")
    video_name = Path(checkpoint_path).stem
    env = await get_env_async(record_video=record_video, video_name=video_name)

    results = []

    try:
        for ep in range(episodes):
            print(f"\n=== Episode {ep + 1}/{episodes} ===")

            obs, info = env.reset()
            total_reward = 0.0
            step = 0

            pbar = tqdm.tqdm(total=max_steps, desc=f"Episode {ep + 1}")

            while step < max_steps:
                # Get action from model
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    action = model(obs_tensor).cpu().numpy().squeeze(0)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                pbar.update(1)

                if verbose and step % 100 == 0:
                    pbar.set_postfix(reward=f"{total_reward:.1f}")

                if terminated or truncated:
                    reason = "collision" if terminated else "truncated"
                    print(f"\nEpisode ended: {reason} at step {step}")
                    break

            pbar.close()

            result = {
                "total_reward": total_reward,
                "steps": step,
                "terminated": terminated,
                "truncated": truncated,
            }
            results.append(result)
            print(f"Episode result: reward={total_reward:.1f}, steps={step}")

    finally:
        env.close()

    # Summary
    if len(results) > 1:
        print("\n=== Summary ===")
        rewards = [r["total_reward"] for r in results]
        steps = [r["steps"] for r in results]
        print(f"Average reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Average steps: {np.mean(steps):.0f} +/- {np.std(steps):.0f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained BC model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/supervised/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="Max steps per episode"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record video of episodes"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable verbose output"
    )
    args = parser.parse_args()

    nest_asyncio.apply()

    asyncio.run(
        run_pretrained_async(
            checkpoint_path=args.checkpoint,
            max_steps=args.max_steps,
            episodes=args.episodes,
            record_video=args.record,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()

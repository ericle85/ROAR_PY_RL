import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import asyncio
import nest_asyncio
import os
from pathlib import Path
from typing import Optional, Dict
import torch as th
from typing import Dict, SupportsFloat, Union
from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList, BaseCallback

RUN_FPS=25
SUBSTEPS_PER_STEP = 5
MODEL_SAVE_FREQ = 50_000
VIDEO_SAVE_FREQ = 20_000
TIME_LIMIT = RUN_FPS * 2 * 60
USE_DISCRETE_ACTIONS = False  # Use continuous actions for SAC
run_name = "SAC_Continuous_RacingLine"

# Racing line path (relative to repo root)
RACING_LINE_PATH = r'C:\Users\shrek\ROAR_PY_RL\racingline\main.npz'

# SAC training parameters
training_params = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,  # replay buffer size
    learning_starts=10_000,  # start training after this many steps
    batch_size=256,
    tau=0.005,  # soft update coefficient
    gamma=0.99,
    train_freq=1,  # update policy every step
    gradient_steps=1,  # gradient steps per update
    ent_coef='auto',  # automatic entropy tuning
    target_update_interval=1,
    target_entropy='auto',
    verbose=1,
    seed=1,
    device='cpu',
)

def find_latest_model(root_path: Path) -> Optional[Path]:
    """
        Find the path of latest model if exists.
    """
    logs_path = (os.path.join(root_path, "logs"))
    if os.path.exists(logs_path) is False:
        print(f"No previous record found in {logs_path}")
        return None
    print(f"logs_path: {logs_path}")
    files = os.listdir(logs_path)
    paths = sorted(files)
    paths_dict: Dict[int, Path] = {int(path.split("_")[2]): path for path in paths}
    if len(paths_dict) == 0:
        return None
    latest_model_file_path: Optional[Path] = Path(os.path.join(logs_path, paths_dict[max(paths_dict.keys())]))
    return latest_model_file_path

def get_env(wandb_run) -> gym.Env:
    env = asyncio.run(initialize_roar_env(
        control_timestep=1.0/RUN_FPS,
        physics_timestep=1.0/(RUN_FPS*SUBSTEPS_PER_STEP),
        racing_line_path=str(RACING_LINE_PATH),
        use_discrete_actions=USE_DISCRETE_ACTIONS
    ))
    env = gym.wrappers.FlattenObservation(env)
    # Only flatten actions for continuous space
    if not USE_DISCRETE_ACTIONS:
        env = FlattenActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = TIME_LIMIT)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f"videos/{wandb_run.name}", step_trigger=lambda x: x % VIDEO_SAVE_FREQ == 0)
    env = Monitor(env, f"logs/{wandb_run.name}_{wandb_run.id}", allow_early_resets=True)
    return env

def main():
    wandb_run = wandb.init(
        project="ROAR_PY_RL",
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    
    env = get_env(wandb_run)

    models_path = f"models/{wandb_run.name}"
    latest_model_path = find_latest_model(Path(models_path))
    latest_model_path = None  # force new model for testing
    
    if latest_model_path is None:
        # create new models
        model = SAC(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{wandb_run.name}",
            **training_params
        )

        # Load pretrained BC weights into actor
        bc_path = r"C:\Users\shrek\ROAR_PY_RL\training\supervised\checkpoints\best_model.pt"
        if os.path.exists(bc_path):
            bc_checkpoint = th.load(bc_path, map_location=model.device, weights_only=True)
            bc_weights = bc_checkpoint["model_state_dict"]

            actor_state = model.actor.state_dict()
            actor_state['latent_pi.0.weight'] = bc_weights['network.0.weight']
            actor_state['latent_pi.0.bias'] = bc_weights['network.0.bias']
            actor_state['latent_pi.2.weight'] = bc_weights['network.2.weight']
            actor_state['latent_pi.2.bias'] = bc_weights['network.2.bias']
            actor_state['mu.weight'] = bc_weights['network.4.weight']
            actor_state['mu.bias'] = bc_weights['network.4.bias']
            model.actor.load_state_dict(actor_state)
            print(f"Loaded pretrained BC weights from {bc_path}")
        else:
            print(f"No pretrained BC weights found at {bc_path}, starting from scratch")
    else:
        # Load the model
        print(f"reloading from {type(latest_model_path)} {latest_model_path}\n\n\n\n")
        model = SAC.load(
            latest_model_path,
            env=env,
            tensorboard_log=f"runs/{wandb_run.name}",
            **training_params
        )

    wandb_callback=WandbCallback(
        gradient_save_freq = MODEL_SAVE_FREQ,
        model_save_path = f"models/{wandb_run.name}",
        verbose = 2,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq = MODEL_SAVE_FREQ,
        verbose = 2,
        save_path = f"{models_path}/logs"
    )
    event_callback = EveryNTimesteps(
        n_steps = MODEL_SAVE_FREQ,
        callback=checkpoint_callback
    )

    callbacks = CallbackList([
        wandb_callback,
        checkpoint_callback, 
        event_callback
    ])

    model.learn(
        total_timesteps=1e7,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    main()

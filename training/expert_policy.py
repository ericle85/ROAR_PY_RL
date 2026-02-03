"""
Expert Policy wrapper that uses the #1 hardcoded ROAR solution.
https://github.com/MightyMark3/ROAR_1_S25

This wraps their RoarCompetitionSolution to work with our training environment.
"""

import sys
import os
import numpy as np
import asyncio
from typing import Tuple, Any, Optional, Dict

# Add the cloned repo to path so we can import their modules
EXPERT_CODE_PATH = os.path.join(os.path.dirname(__file__), "..", "ROAR_1_S25", "competition_code")
sys.path.insert(0, EXPERT_CODE_PATH)

from submission import RoarCompetitionSolution


class ExpertPolicyWrapper:
    """
    Wraps the #1 hardcoded ROAR solution to work as an expert policy.

    This class takes the raw (unwrapped) environment and extracts the sensors
    needed by the hardcoded solution.
    """

    def __init__(self, env):
        """
        Initialize the expert policy from an environment.

        Args:
            env: The gymnasium environment (can be wrapped, will unwrap to get sensors)
        """
        self.env = env
        self.expert = None
        self._initialized = False

        # Get the unwrapped environment to access sensors
        self.unwrapped_env = env.unwrapped

    def _get_sensors(self):
        """Extract sensors from the unwrapped environment."""
        env = self.unwrapped_env

        # Access sensors directly from the environment
        return {
            'location_sensor': env.location_sensor,
            'velocity_sensor': env.velocimeter_sensor,
            'rpy_sensor': env.roll_pitch_yaw_sensor,
            'collision_sensor': env.collision_sensor,
        }

    async def initialize(self):
        """Initialize the expert solution (must be called before predict)."""
        if self._initialized:
            return

        env = self.unwrapped_env
        sensors = self._get_sensors()

        # Create the competition solution
        # Set apply_action=False so we compute control but don't apply it
        # (the env.step() will apply our converted action instead)
        self.expert = RoarCompetitionSolution(
            maneuverable_waypoints=env.waypoints_tracer._waypoints,  # Will be overwritten by initialize()
            vehicle=env.roar_py_actor,
            camera_sensor=None,
            location_sensor=sensors['location_sensor'],
            velocity_sensor=sensors['velocity_sensor'],
            rpy_sensor=sensors['rpy_sensor'],
            occupancy_map_sensor=None,
            collision_sensor=sensors['collision_sensor'],
            apply_action=False,  # Don't apply action - let env.step() handle it
        )

        # Initialize loads the custom waypoints and sets up section tracking
        await self.expert.initialize()
        self._initialized = True

        print(f"Expert policy initialized at waypoint {self.expert.current_waypoint_idx}")

    def initialize_sync(self):
        """Synchronous wrapper for initialize()."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a task
            import nest_asyncio
            nest_asyncio.apply()
        asyncio.run(self.initialize())

    async def get_action_async(self) -> Dict[str, float]:
        """
        Get the expert action for the current state (async version).

        Returns:
            Dict with 'throttle', 'steer', 'brake' keys
        """
        if not self._initialized:
            await self.initialize()

        # Call the expert's step method
        control = await self.expert.step()
        return control

    def get_action(self) -> Dict[str, float]:
        """
        Get the expert action for the current state (sync version).

        Returns:
            Dict with 'throttle', 'steer', 'brake' keys
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new task in the existing loop
            import nest_asyncio
            nest_asyncio.apply()
        return asyncio.run(self.get_action_async())

    def predict(
        self,
        observation: Any,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        SB3-compatible predict interface.

        Note: The observation is ignored - the expert reads directly from sensors.

        Args:
            observation: Ignored (expert uses raw sensors)
            deterministic: Ignored (expert is always deterministic)

        Returns:
            Tuple of (action_array, None)
            action_array: [throttle, steer] for FlattenActionWrapper
        """
        control = self.get_action()

        # Convert to our action format
        # Our SimplifyCarlaActionFilter expects throttle in [-1, 1] where negative = brake
        throttle = float(control.get('throttle', 0.0))
        brake = float(control.get('brake', 0.0))
        steer = float(control.get('steer', 0.0))

        # Combine throttle and brake into single value
        # positive = throttle, negative = brake
        if brake > throttle:
            combined_throttle = -brake
        else:
            combined_throttle = throttle

        # NOTE: FlattenActionWrapper unflattens Dict keys alphabetically
        # "steer" < "throttle", so flattened order is [steer, throttle]
        action = np.array([steer, combined_throttle], dtype=np.float32)
        return action, None


class ExpertRunner:
    """
    Helper class to run the expert policy in an environment and collect data.
    """

    def __init__(self, env, expert: ExpertPolicyWrapper):
        self.env = env
        self.expert = expert

    async def run_episode(
        self,
        max_steps: int = 3000,
        collect_data: bool = False,
        verbose: bool = True
    ) -> dict:
        """
        Run a single episode with the expert.

        Args:
            max_steps: Maximum steps per episode
            collect_data: Whether to collect (obs, action) pairs
            verbose: Print progress

        Returns:
            Dict with episode stats and optionally collected data
        """
        # Reset env FIRST to place vehicle at spawn point
        obs, info = self.env.reset()

        # THEN initialize expert so it finds waypoints from the correct location
        self.expert._initialized = False  # Force re-initialization
        await self.expert.initialize()

        collected_obs = []
        collected_actions = []
        total_reward = 0.0

        for step in range(max_steps):
            # Get expert action (reads from sensors directly)
            control = await self.expert.get_action_async()

            # Convert to env action format
            throttle = float(control.get('throttle', 0.0))
            brake = float(control.get('brake', 0.0))
            steer = float(control.get('steer', 0.0))

            if brake > throttle:
                combined_throttle = -brake
            else:
                combined_throttle = throttle

            # NOTE: FlattenActionWrapper unflattens Dict keys alphabetically
            # "steer" < "throttle", so flattened order is [steer, throttle]
            action = np.array([steer, combined_throttle], dtype=np.float32)

            if collect_data:
                collected_obs.append(obs.copy())
                collected_actions.append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if verbose and step % 100 == 0:
                speed = self.expert.expert.velocity_sensor.get_last_gym_observation()
                speed_kmh = np.linalg.norm(speed) * 3.6
                print(f"Step {step}: reward={total_reward:.1f}, speed={speed_kmh:.1f} km/h")

            if terminated or truncated:
                if verbose:
                    print(f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
                break

        result = {
            'total_reward': total_reward,
            'steps': step + 1,
            'terminated': terminated,
            'truncated': truncated,
        }

        if collect_data:
            result['observations'] = np.array(collected_obs)
            result['actions'] = np.array(collected_actions)

        return result

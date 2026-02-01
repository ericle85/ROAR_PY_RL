import gymnasium as gym
import numpy as np
from typing import Any, List, Optional, SupportsFloat, Tuple, Dict, Union
import roar_py_rl_carla
import roar_py_carla
import roar_py_interface
import carla


def build_racing_action_lut() -> np.ndarray:
    """
    Build a lookup table of (throttle, steer) pairs for racing.
    Similar to RLGym's approach - curated action combinations.

    Returns array of shape (N, 2) where each row is [throttle, steer].
    """
    actions = []

    # Throttle values: focus on forward driving with some braking
    throttle_values = [-1.0, -0.5, 0.0, 0.3, 0.5, 0.7, 0.85, 1.0]

    # Steer values: finer resolution near center for precise control
    steer_values = [-1.0, -0.7, -0.5, -0.3, -0.15, -0.07, 0.0, 0.07, 0.15, 0.3, 0.5, 0.7, 1.0]

    for throttle in throttle_values:
        for steer in steer_values:
            actions.append([throttle, steer])

    return np.array(actions, dtype=np.float32)


# Pre-built LUT for racing actions
RACING_ACTION_LUT = build_racing_action_lut()


class DiscreteToContActionWrapper(gym.ActionWrapper):
    """
    Converts a single discrete action index to (throttle, steer) via lookup table.
    Similar to RLGym's approach with curated action combinations.

    Action space: Discrete(N) where N = number of action combinations in LUT.
    """
    def __init__(self, env: gym.Env, action_lut: np.ndarray = None):
        super().__init__(env)
        self.lut = action_lut if action_lut is not None else RACING_ACTION_LUT
        self._action_space = gym.spaces.Discrete(len(self.lut))
        print(f"Discrete action space with {len(self.lut)} actions")

    def action(self, action: int) -> Dict[str, np.ndarray]:
        """Convert discrete index to continuous Dict action via LUT."""
        throttle, steer = self.lut[action]
        return {
            "throttle": np.array([throttle], dtype=np.float32),
            "steer": np.array([steer], dtype=np.float32)
        }


class SimplifyCarlaActionFilter(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_space = gym.spaces.Dict({
            "throttle": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
            "steer": gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        })
    
    def action(self, action: Dict[str, Union[SupportsFloat, float]]) -> Dict[str, Union[SupportsFloat, float]]:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        # action = {
        #     "throttle": [-1.0, 1.0],
        #     "steer": [-1.0, 1.0]
        # }
        # if isinstance(self.env.unwrapped, roar_py_rl_carla.RoarRLCarlaSimEnv):
        #     velocity = self.env.unwrapped.roar_py_actor.get_linear_3d_velocity()
        #     velocity = np.linalg.norm(velocity)
        
        real_action = {
            "throttle": np.clip(action["throttle"], 0.0, 1.0),
            "brake": np.clip(-action["throttle"], 0.0, 1.0),
            "steer": action["steer"],
            "hand_brake": 0.0,
            "reverse": 0.0
        }
        return real_action

async def initialize_roar_env(
    carla_host : str = "localhost",
    carla_port : int = 2000,
    control_timestep : float = 0.05,
    physics_timestep : float = 0.01,
    waypoint_information_distances : list = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 80.0, 100.0],
    image_width : int = 400,
    image_height : int = 200,
    racing_line_path : str = None,
    use_discrete_actions : bool = False,
    action_lut : np.ndarray = None
):
    carla_client = carla.Client(carla_host, carla_port)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(control_timestep, physics_timestep)
    world.set_asynchronous(False)
    await world.step()
    roar_py_instance.clean_actors_not_registered(["vehicle.*", "sensor.*"])

    spawn_point = world.spawn_points[0]

    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point[0] + np.array([0, 0, 2.0]),
        spawn_point[1],
        True,
        "vehicle"
    )
    assert vehicle is not None, "Failed to spawn vehicle"
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3),
        name="collision_sensor"
    )
    assert collision_sensor is not None, "Failed to attach collision sensor"
    
    #TODO: Attach next waypoint to observation
    velocimeter_sensor = vehicle.attach_velocimeter_sensor("velocimeter")
    local_velocimeter_sensor = vehicle.attach_local_velocimeter_sensor("local_velocimeter")

    location_sensor = vehicle.attach_location_in_world_sensor("location")
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor("roll_pitch_yaw")
    gyroscope_sensor = vehicle.attach_gyroscope_sensor("gyroscope")
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=image_width,
        image_height=image_height
    )
    # occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
    #     50,
    #     50,
    #     4.0,
    #     4.0,
    #     name="occupancy_map"
    # )

    await world.step()
    await vehicle.receive_observation()
    env = roar_py_rl_carla.RoarRLCarlaSimEnv(
        vehicle,
        world.maneuverable_waypoints,
        location_sensor,
        rpy_sensor,
        velocimeter_sensor,
        collision_sensor,
        waypoint_information_distances=set(waypoint_information_distances),
        world = world,
        collision_threshold = 1.0,
        racing_line_path = racing_line_path
    )
    env = SimplifyCarlaActionFilter(env)
    env = gym.wrappers.FilterObservation(env, ["gyroscope", "waypoints_information", "local_velocimeter"])

    if use_discrete_actions:
        env = DiscreteToContActionWrapper(env, action_lut=action_lut)

    return env
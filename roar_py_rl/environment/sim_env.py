from typing import List, Optional
from roar_py_interface import RoarPyActor, RoarPySensor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor, RoarPyRollPitchYawSensor, RoarPyWaypointsTracker, RoarPyWaypointsProjection
from .base_env import RoarRLEnv
from typing import Any, Dict, SupportsFloat, Tuple, Optional, Set
import gymnasium as gym
import numpy as np
from shapely import Polygon, Point
from collections import OrderedDict
from pathlib import Path
from scipy.spatial import KDTree

def distance_to_waypoint_polygon(
    waypoint_1: RoarPyWaypoint,
    waypoint_2: RoarPyWaypoint,
    point: np.ndarray
):
    p1, p2 = waypoint_1.line_representation
    p3, p4 = waypoint_2.line_representation
    polygon = Polygon([p1, p2, p4, p3])
    return polygon.distance(Point(point))

def global_to_local(
    global_point: np.ndarray,
    local_origin: np.ndarray,
    local_yaw: float
) -> np.ndarray:
    delta_global = global_point - local_origin
    delta_local = np.array([
        np.cos(local_yaw) * delta_global[0] + np.sin(local_yaw) * delta_global[1],
        -np.sin(local_yaw) * delta_global[0] + np.cos(local_yaw) * delta_global[1]
    ])
    return delta_local

def normalize_rad(rad : float) -> float:
    return (rad % (2 * np.pi) + 3 * np.pi) % (2 * np.pi) - np.pi


class RacingLineTracker:
    """Tracks vehicle position relative to an optimized racing line."""

    def __init__(self, racing_line_path: str):
        data = np.load(racing_line_path)
        self.locations = data['locations'][:, :2]  # Only x, y for 2D distance
        self.locations_3d = data['locations']
        self.rotations = data['rotations']
        self.lane_widths = data['lane_widths']

        # Build KD-tree for fast nearest neighbor lookup
        self.kdtree = KDTree(self.locations)
        self.num_points = len(self.locations)

        # Precompute cumulative distances along racing line
        diffs = np.diff(self.locations, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.cumulative_distances = np.zeros(self.num_points)
        self.cumulative_distances[1:] = np.cumsum(segment_lengths)
        self.total_length = self.cumulative_distances[-1]

        self._last_idx = 0

    def get_nearest_point(self, location: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Find nearest racing line point to given location.
        Returns: (index, distance, nearest_point)
        """
        loc_2d = location[:2]
        dist, idx = self.kdtree.query(loc_2d)
        self._last_idx = idx
        return idx, dist, self.locations_3d[idx]

    def get_progress(self, idx: int) -> float:
        """Get progress along racing line (0 to total_length)."""
        return self.cumulative_distances[idx]

    def get_delta_progress(self, old_idx: int, new_idx: int) -> float:
        """
        Calculate progress made between two indices.
        Handles wraparound for closed tracks.
        """
        delta = self.cumulative_distances[new_idx] - self.cumulative_distances[old_idx]

        # Handle wraparound: if delta is very negative, we crossed the start line
        if delta < -self.total_length / 2:
            delta += self.total_length
        # If delta is very positive (going backwards across start), make it negative
        elif delta > self.total_length / 2:
            delta -= self.total_length

        return delta


class RoarRLSimEnv(RoarRLEnv):
    def __init__(
            self,
            actor: RoarPyActor,
            manuverable_waypoints: List[RoarPyWaypoint],
            location_sensor : RoarPyLocationInWorldSensor,
            roll_pitch_yaw_sensor : RoarPyRollPitchYawSensor,
            velocimeter_sensor : RoarPyVelocimeterSensor,
            collision_sensor : RoarPyCollisionSensor,
            collision_threshold : float = 30.0,
            waypoint_information_distances : Set[float] = set([]),
            world: Optional[RoarPyWorld] = None,
            render_mode="rgb_array",
            racing_line_path: Optional[str] = None
        ) -> None:
        super().__init__(actor, manuverable_waypoints, world, render_mode)
        self.location_sensor = location_sensor
        self.roll_pitch_yaw_sensor = roll_pitch_yaw_sensor
        self.velocimeter_sensor = velocimeter_sensor
        self.collision_sensor = collision_sensor
        self.collision_threshold = collision_threshold
        self.waypoint_information_distances = waypoint_information_distances

        self.waypoints_tracer = RoarPyWaypointsTracker(manuverable_waypoints)
        self._traced_projection : RoarPyWaypointsProjection = RoarPyWaypointsProjection(0,0.0)
        self._delta_distance_travelled = 0.0

        # Racing line tracking
        self.racing_line: Optional[RacingLineTracker] = None
        self._racing_line_idx = 0
        self._racing_line_delta_progress = 0.0
        if racing_line_path is not None:
            self.racing_line = RacingLineTracker(racing_line_path)
            print(f"Loaded racing line with {self.racing_line.num_points} points, total length: {self.racing_line.total_length:.1f}m")

    @property
    def observation_space(self) -> gym.Space:
        space = super().observation_space
        if len(self.waypoint_information_distances) > 0:
            waypoints_info_space_dict = OrderedDict()
            for dist in sorted(self.waypoint_information_distances):
                waypoints_info_space_dict[f"waypoint_{dist}"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,), # x, y, yaw, lane_width
                    dtype=np.float32
                )
            space["waypoints_information"] = gym.spaces.Dict(waypoints_info_space_dict)
        
        return space

    def observation(self, info_dict : Dict[str, Any]) -> Dict[str, Any]:
        obs = super().observation(info_dict)

        location = self.location_sensor.get_last_gym_observation()
        yaw = self.roll_pitch_yaw_sensor.get_last_gym_observation()[2]

        if len(self.waypoint_information_distances) > 0:
            waypoint_info = {}
            for trace_dist in self.waypoint_information_distances:
                traced_projection = self.waypoints_tracer.trace_forward_projection(self._traced_projection, trace_dist)
                traced_projection_wp = self.waypoints_tracer.get_interpolated_waypoint(traced_projection)
                waypoint_info[f"waypoint_{trace_dist}"] = np.concatenate([
                    global_to_local(traced_projection_wp.location, location, yaw),
                    np.array([normalize_rad(traced_projection_wp.roll_pitch_yaw[2] - yaw)]),
                    np.array([traced_projection_wp.lane_width])
                ])

            obs["waypoints_information"] = waypoint_info
        info_dict["delta_distance_travelled"] = self._delta_distance_travelled
        return obs

    def reset_vehicle(self) -> None:
        return NotImplementedError

    @property
    def sensors_to_update(self) -> List[RoarPySensor]:
        return [
            sensor for sensor in
            [self.location_sensor, self.roll_pitch_yaw_sensor, self.velocimeter_sensor, self.collision_sensor]
            if sensor not in self.roar_py_actor.get_sensors()
        ]

    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)

        if collision_impulse_norm > self.collision_threshold:
            return 0  # Terminal collision

        # Use racing line if available, otherwise fall back to waypoint centerline
        if self.racing_line is not None:
            dist_to_line = self._racing_line_dist
            delta_progress = self._racing_line_delta_progress

            # Reward = progress along racing line, penalized by distance from it
            # Gaussian-like penalty: exp(-dist^2 / (2 * sigma^2)), sigma ~= 2m
            proximity_factor = np.exp(-(dist_to_line ** 2) / 8.0)

            if delta_progress <= 0:
                # Going backwards: penalize more heavily
                reward = delta_progress * 10.0 * (1.0 - 0.5 * proximity_factor)
            else:
                # Going forwards: reward scaled by proximity to racing line
                reward = delta_progress * 10.0 * (0.5 + 0.5 * proximity_factor)

            return reward
        else:
            # Fallback: original waypoint-based reward
            dist_to_projection = np.linalg.norm(
                self.location_sensor.get_last_gym_observation() - self._traced_projection_point.location
            )
            if self._delta_distance_travelled <= 0:
                normalized_rew = self._delta_distance_travelled * 10.0 * (0.2 * dist_to_projection + 1.0)
            else:
                normalized_rew = self._delta_distance_travelled * 10.0 / (0.2 * dist_to_projection + 1.0)
            return normalized_rew
    
    def _perform_waypoint_trace(self, location: Optional[np.ndarray] = None) -> None:
        if location is None:
            location = self.location_sensor.get_last_gym_observation()
        _last_traced_projection = self._traced_projection
        self._traced_projection = self.waypoints_tracer.trace_point(location, _last_traced_projection.waypoint_idx)
        self._traced_projection_point = self.waypoints_tracer.get_interpolated_waypoint(self._traced_projection)
        self._delta_distance_travelled = self.waypoints_tracer.delta_distance_projection(_last_traced_projection, self._traced_projection)

        # Track racing line if available
        if self.racing_line is not None:
            old_idx = self._racing_line_idx
            new_idx, dist, _ = self.racing_line.get_nearest_point(location)
            self._racing_line_idx = new_idx
            self._racing_line_dist = dist
            self._racing_line_delta_progress = self.racing_line.get_delta_progress(old_idx, new_idx)

    def _step(self, action: Any) -> None:
        self._perform_waypoint_trace()

    def _reset(self) -> None:
        # Initialize racing line tracking before waypoint trace
        if self.racing_line is not None:
            location = self.location_sensor.get_last_gym_observation()
            idx, dist, _ = self.racing_line.get_nearest_point(location)
            self._racing_line_idx = idx
            self._racing_line_dist = dist
            self._racing_line_delta_progress = 0.0

        self._perform_waypoint_trace()
        self._delta_distance_travelled = 0.0

    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm > 0:
            print(f"Collision detected with impulse {collision_impulse_norm}", self.collision_sensor.get_last_observation().impulse_normals)
        if collision_impulse_norm > self.collision_threshold:
            print("Terminated due to collision")
            return True
        
        return False

    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        return False

from typing import List, Optional, NamedTuple
from roar_py_interface import RoarPyActor, RoarPySensor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor, RoarPyRollPitchYawSensor, RoarPyWaypointsTracker, RoarPyWaypointsProjection
from .base_env import RoarRLEnv
from typing import Any, Dict, SupportsFloat, Tuple, Optional, Set
import gymnasium as gym
import numpy as np
from shapely import Polygon, Point, LineString
from collections import OrderedDict
from pathlib import Path
import math


class RacingLineProjection(NamedTuple):
    """Projection result onto racing line segment."""
    segment_idx: int  # Index of the segment start point
    distance_along_segment: float  # Distance from segment start to projected point

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
    """
    Tracks vehicle position relative to an optimized racing line.
    Uses segment projection similar to RoarPyWaypointsTracker for accurate distance calculation.
    """

    def __init__(self, racing_line_path: str):
        data = np.load(racing_line_path)
        self.locations = data['locations'][:, :2]  # x, y for 2D projection
        self.locations_3d = data['locations']
        self.rotations = data['rotations']
        self.lane_widths = data['lane_widths']
        self.num_points = len(self.locations)

        # Precompute segment lengths and cumulative distances
        self._segment_lengths = np.zeros(self.num_points)
        for i in range(self.num_points):
            next_i = (i + 1) % self.num_points
            self._segment_lengths[i] = np.linalg.norm(
                self.locations[next_i] - self.locations[i]
            )

        self._cumulative_distances = np.zeros(self.num_points)
        for i in range(1, self.num_points):
            self._cumulative_distances[i] = (
                self._cumulative_distances[i - 1] + self._segment_lengths[i - 1]
            )

        # Total length includes the closing segment back to start
        self._total_length = self._cumulative_distances[-1] + self._segment_lengths[-1]

    @property
    def total_length(self) -> float:
        return self._total_length

    def _point_to_segment_distance(
        self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate distance from point to line segment and projection distance along segment.
        Returns: (perpendicular_distance, distance_along_segment)
        """
        seg_vec = seg_end - seg_start
        seg_len = np.linalg.norm(seg_vec)

        if seg_len < 1e-9:
            return np.linalg.norm(point - seg_start), 0.0

        # Project point onto line, get parameter t
        t = np.dot(point - seg_start, seg_vec) / (seg_len * seg_len)
        t_clamped = np.clip(t, 0.0, 1.0)

        # Closest point on segment
        closest = seg_start + t_clamped * seg_vec
        dist = np.linalg.norm(point - closest)

        # Distance along segment (can be negative or > seg_len if outside segment)
        dist_along = t * seg_len

        return dist, dist_along

    def trace_point(
        self, point: np.ndarray, start_idx: int = 0
    ) -> RacingLineProjection:
        """
        Trace a point to find the closest segment on the racing line.
        Searches forward and backward from start_idx for efficiency.
        Returns: RacingLineProjection with segment index and distance along segment.
        """
        point_2d = point[:2]
        min_dist = float('inf')
        best_idx = start_idx
        best_dist_along = 0.0

        # Search forward and backward from start_idx
        search_range = int(math.ceil(self.num_points / 2)) + 2

        for i in range(1, search_range):
            # Forward segment
            fwd_idx = (start_idx + i - 1) % self.num_points
            fwd_next = (start_idx + i) % self.num_points
            fwd_dist, fwd_along = self._point_to_segment_distance(
                point_2d, self.locations[fwd_idx], self.locations[fwd_next]
            )

            if fwd_dist < min_dist:
                min_dist = fwd_dist
                best_idx = fwd_idx
                best_dist_along = np.clip(fwd_along, 0.0, self._segment_lengths[fwd_idx])

            # Backward segment
            bwd_idx = (start_idx - i) % self.num_points
            bwd_next = (start_idx - i + 1) % self.num_points
            bwd_dist, bwd_along = self._point_to_segment_distance(
                point_2d, self.locations[bwd_idx], self.locations[bwd_next]
            )

            if bwd_dist < min_dist:
                min_dist = bwd_dist
                best_idx = bwd_idx
                best_dist_along = np.clip(bwd_along, 0.0, self._segment_lengths[bwd_idx])

            # Early exit if we found an exact match
            if min_dist < 1e-6:
                break

        return RacingLineProjection(best_idx, best_dist_along)

    def total_distance_from_start(self, projection: RacingLineProjection) -> float:
        """Get total distance from racing line start to this projection."""
        return self._cumulative_distances[projection.segment_idx] + projection.distance_along_segment

    def delta_distance_projection(
        self,
        origin: RacingLineProjection,
        destination: RacingLineProjection
    ) -> float:
        """
        Calculate signed distance traveled between two projections.
        Positive = forward progress, negative = backward.
        Handles wraparound for closed tracks.
        """
        dist_origin = self.total_distance_from_start(origin)
        dist_destination = self.total_distance_from_start(destination)

        delta = (dist_destination - dist_origin + self._total_length) % self._total_length

        # If delta > half the track, we went backwards
        if delta > self._total_length / 2:
            delta -= self._total_length

        return delta

    def get_interpolated_location(self, projection: RacingLineProjection) -> np.ndarray:
        """Get the exact 3D location on the racing line at this projection."""
        idx = projection.segment_idx
        next_idx = (idx + 1) % self.num_points
        seg_len = self._segment_lengths[idx]

        if seg_len < 1e-9:
            return self.locations_3d[idx].copy()

        alpha = np.clip(projection.distance_along_segment / seg_len, 0.0, 1.0)
        return (1 - alpha) * self.locations_3d[idx] + alpha * self.locations_3d[next_idx]

    def get_distance_to_racing_line(
        self, point: np.ndarray, projection: RacingLineProjection
    ) -> float:
        """Get perpendicular distance from point to the racing line at projection."""
        interpolated = self.get_interpolated_location(projection)
        return np.linalg.norm(point[:2] - interpolated[:2])

    def get_signed_lateral_offset(
        self, point: np.ndarray, projection: RacingLineProjection
    ) -> float:
        """Get signed lateral offset from centerline.

        Positive = right of centerline, Negative = left of centerline.
        Uses cross product to determine side.
        """
        # Get the interpolated point on the racing line
        line_point = self.get_interpolated_location(projection)[:2]

        # Get the direction of the racing line at this point (tangent)
        idx = projection.segment_idx
        next_idx = (idx + 1) % self.num_points
        line_dir = self.locations[next_idx] - self.locations[idx]
        line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-9)

        # Vector from line point to vehicle
        to_vehicle = point[:2] - line_point

        # Cross product in 2D: positive = vehicle is to the right
        # cross = line_dir.x * to_vehicle.y - line_dir.y * to_vehicle.x
        cross = line_dir[0] * to_vehicle[1] - line_dir[1] * to_vehicle[0]

        # Distance with sign
        distance = np.linalg.norm(to_vehicle)
        return float(np.sign(cross) * distance)

    def trace_forward_projection(
        self, projection: RacingLineProjection, distance: float
    ) -> RacingLineProjection:
        """
        Trace forward along the racing line by a given distance.
        Returns a new projection at that point.
        """
        current_idx = projection.segment_idx
        remaining_dist = distance + projection.distance_along_segment

        # Walk forward through segments until we've covered the distance
        while remaining_dist > self._segment_lengths[current_idx]:
            remaining_dist -= self._segment_lengths[current_idx]
            current_idx = (current_idx + 1) % self.num_points

        return RacingLineProjection(current_idx, remaining_dist)

    def get_interpolated_yaw(self, projection: RacingLineProjection) -> float:
        """Get interpolated yaw angle at this projection point."""
        idx = projection.segment_idx
        next_idx = (idx + 1) % self.num_points
        seg_len = self._segment_lengths[idx]

        if seg_len < 1e-9:
            return self.rotations[idx][2]  # yaw is third element (roll, pitch, yaw)

        alpha = np.clip(projection.distance_along_segment / seg_len, 0.0, 1.0)

        # Interpolate yaw with angle wrapping
        yaw1 = self.rotations[idx][2]
        yaw2 = self.rotations[next_idx][2]
        delta_yaw = normalize_rad(yaw2 - yaw1)
        return normalize_rad(yaw1 + alpha * delta_yaw)

    def get_interpolated_lane_width(self, projection: RacingLineProjection) -> float:
        """Get interpolated lane width at this projection point."""
        idx = projection.segment_idx
        next_idx = (idx + 1) % self.num_points
        seg_len = self._segment_lengths[idx]

        if seg_len < 1e-9:
            return self.lane_widths[idx]

        alpha = np.clip(projection.distance_along_segment / seg_len, 0.0, 1.0)
        return (1 - alpha) * self.lane_widths[idx] + alpha * self.lane_widths[next_idx]


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
            racing_line_path: Optional[str] = None,
            centerline_path: Optional[str] = None
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
        self._racing_line_projection: Optional[RacingLineProjection] = None
        self._racing_line_dist = 0.0
        self._racing_line_delta_progress = 0.0
        if racing_line_path is not None:
            self.racing_line = RacingLineTracker(racing_line_path)
            self._racing_line_projection = RacingLineProjection(0, 0.0)
            print(f"Loaded racing line with {self.racing_line.num_points} points, total length: {self.racing_line.total_length:.1f}m")

        # Centerline tracking (for lateral offset / wall awareness)
        self.centerline: Optional[RacingLineTracker] = None
        self._centerline_projection: Optional[RacingLineProjection] = None
        if centerline_path is not None:
            self.centerline = RacingLineTracker(centerline_path)
            self._centerline_projection = RacingLineProjection(0, 0.0)
            print(f"Loaded centerline with {self.centerline.num_points} points, total length: {self.centerline.total_length:.1f}m")

        # Previous action for observation (throttle, steer)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._action_smoothness_penalty = 0.0

        # Lateral offset from centerline (signed: + = right, - = left)
        self._lateral_offset = 0.0

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

        # Previous action (throttle, steer)
        space["prev_action"] = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Lateral offset from centerline (signed: + = right, - = left)
        space["lateral_offset"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )

        return space

    def observation(self, info_dict : Dict[str, Any]) -> Dict[str, Any]:
        obs = super().observation(info_dict)

        location = self.location_sensor.get_last_gym_observation()
        yaw = self.roll_pitch_yaw_sensor.get_last_gym_observation()[2]

        if len(self.waypoint_information_distances) > 0:
            waypoint_info = {}
            for trace_dist in self.waypoint_information_distances:
                # Use racing line waypoints if available, otherwise fall back to centerline
                if self.racing_line is not None:
                    traced_projection = self.racing_line.trace_forward_projection(
                        self._racing_line_projection, trace_dist
                    )
                    wp_location = self.racing_line.get_interpolated_location(traced_projection)
                    wp_yaw = self.racing_line.get_interpolated_yaw(traced_projection)
                    wp_lane_width = self.racing_line.get_interpolated_lane_width(traced_projection)
                    waypoint_info[f"waypoint_{trace_dist}"] = np.concatenate([
                        global_to_local(wp_location, location, yaw),
                        np.array([normalize_rad(wp_yaw - yaw)]),
                        np.array([wp_lane_width])
                    ])
                else:
                    traced_projection = self.waypoints_tracer.trace_forward_projection(
                        self._traced_projection, trace_dist
                    )
                    traced_projection_wp = self.waypoints_tracer.get_interpolated_waypoint(traced_projection)
                    waypoint_info[f"waypoint_{trace_dist}"] = np.concatenate([
                        global_to_local(traced_projection_wp.location, location, yaw),
                        np.array([normalize_rad(traced_projection_wp.roll_pitch_yaw[2] - yaw)]),
                        np.array([traced_projection_wp.lane_width])
                    ])

            obs["waypoints_information"] = waypoint_info

        obs["prev_action"] = self._prev_action.copy()
        obs["lateral_offset"] = np.array([self._lateral_offset], dtype=np.float32)
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

            # Decoupled reward: velocity + line following + progress (additive, not multiplicative)
            # This prevents the model from aggressively diving toward the racing line

            # 1. Velocity reward - primary incentive to go fast
            velocity = np.linalg.norm(self.velocimeter_sensor.get_last_gym_observation())
            velocity_reward = velocity * 0.1

            # 2. Distance penalty - gentle nudge toward racing line (not multiplied by progress)
            line_penalty = -dist_to_line * 0.3

            # 3. Small progress bonus - reward forward movement
            progress_reward = delta_progress * 5.0 if delta_progress > 0 else delta_progress * 10.0

            # 4. Smoothness penalty - penalize jerky steering
            smoothness_penalty = -0.1 * self._action_smoothness_penalty

            reward = velocity_reward + line_penalty + progress_reward + smoothness_penalty
            return reward
        else:
            # Fallback: original waypoint-based reward
            dist_to_projection = np.linalg.norm(
                self.location_sensor.get_last_gym_observation() - self._traced_projection_point.location
            )
            if self._delta_distance_travelled <= 0:
                normalized_rew = self._delta_distance_travelled * 20.0 * (0.2 * dist_to_projection + 1.0)
            else:
                normalized_rew = self._delta_distance_travelled * 20.0 / (0.2 * dist_to_projection + 1.0)
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
            old_projection = self._racing_line_projection
            new_projection = self.racing_line.trace_point(location, old_projection.segment_idx)
            self._racing_line_projection = new_projection
            self._racing_line_dist = self.racing_line.get_distance_to_racing_line(location, new_projection)
            self._racing_line_delta_progress = self.racing_line.delta_distance_projection(old_projection, new_projection)

        # Track centerline for lateral offset (wall awareness)
        if self.centerline is not None:
            old_centerline_proj = self._centerline_projection
            new_centerline_proj = self.centerline.trace_point(location, old_centerline_proj.segment_idx)
            self._centerline_projection = new_centerline_proj
            self._lateral_offset = self.centerline.get_signed_lateral_offset(location, new_centerline_proj)
        else:
            self._lateral_offset = 0.0

    def _step(self, action: Any) -> None:
        # Extract throttle/steer from dict
        if isinstance(action, dict):
            throttle = float(action.get("throttle", 0.0))
            steer = float(action.get("steer", 0.0))
            # Handle array values
            if hasattr(throttle, "__len__"):
                throttle = float(throttle[0]) if len(throttle) > 0 else 0.0
            if hasattr(steer, "__len__"):
                steer = float(steer[0]) if len(steer) > 0 else 0.0
            current_action = np.array([throttle, steer], dtype=np.float32)

            # Compute steering smoothness penalty BEFORE updating prev_action
            steer_change = current_action[1] - self._prev_action[1]
            self._action_smoothness_penalty = steer_change ** 2

            # Now update prev_action for next step
            self._prev_action = current_action
        self._perform_waypoint_trace()

    def _reset(self) -> None:
        location = self.location_sensor.get_last_gym_observation()

        # Initialize racing line tracking before waypoint trace
        if self.racing_line is not None:
            self._racing_line_projection = self.racing_line.trace_point(location, 0)
            self._racing_line_dist = self.racing_line.get_distance_to_racing_line(
                location, self._racing_line_projection
            )
            self._racing_line_delta_progress = 0.0

        # Initialize centerline tracking for lateral offset
        if self.centerline is not None:
            self._centerline_projection = self.centerline.trace_point(location, 0)
            self._lateral_offset = self.centerline.get_signed_lateral_offset(
                location, self._centerline_projection
            )
        else:
            self._lateral_offset = 0.0

        self._perform_waypoint_trace()
        self._delta_distance_travelled = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._action_smoothness_penalty = 0.0

    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        collision_impulse : np.ndarray = self.collision_sensor.get_last_gym_observation()
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm > 0:
            print(f"Collision detected with impulse {collision_impulse_norm}")
        if collision_impulse_norm > self.collision_threshold:
            print("Terminated due to collision")
            return True
        
        return False

    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        return False

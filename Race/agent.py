import carla
import time
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from heapq import heappush, heappop

def is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y, front_angle_threshold=10):
    """
    Check if an obstacle is in front of the ego vehicle within a specified angular range.
    """
    ego_yaw_rad = math.radians(ego_yaw)
    forward_vector = np.array([math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)])
    obstacle_vector = np.array([obstacle_x - ego_x, obstacle_y - ego_y])
    obstacle_distance = np.linalg.norm(obstacle_vector)

    if obstacle_distance > 0:
        obstacle_vector /= obstacle_distance

    dot_product = np.dot(forward_vector, obstacle_vector)
    angle_to_obstacle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
    # print("Obstacle Detected")
    return angle_to_obstacle < front_angle_threshold



class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.step_size = 0.5
        self.max_iterations = 2000
        self.velocity_history = []  # Store (time, velocity)
        self.boundary = []
        self.step = 0

    def log_velocity(self, time, velocity):
        """
        Log the velocity at a given time.
        """
        self.velocity_history.append((time, velocity))

    def plot_velocity_diagram(self):
        """
        Plot a velocity diagram based on the logged velocities.
        """
        if not self.velocity_history:
            print("No velocity data to plot.")
            return
        
        times, velocities = zip(*self.velocity_history)
        plt.figure(figsize=(10, 6))
        plt.plot(times, velocities, label='Velocity (m/s)', linewidth=2)
        plt.title('Velocity vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Dynamically calculate the trajectory and control vehicle movement.
        Use A* if an obstacle is detected; otherwise, follow normal trajectory or slow down only if the obstacle blocks the trajectory.
        """
        control = carla.VehicleControl()
        self.boundary = boundary

        # 1. Get the current position and state of the Ego vehicle
        ego_location = transform.location
        ego_x, ego_y = ego_location.x, ego_location.y
        ego_yaw = transform.rotation.yaw
        ego_vel = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        current_time = len(self.velocity_history) * self.step_size / 2
        self.log_velocity(current_time, ego_vel)

        # 2. Calculate the local trajectory
        trajectory = self.compute_local_trajectory(boundary)

        # 3. Find the trajectory target point
        closest_idx = min(
            range(len(trajectory)),
            key=lambda i: math.sqrt((trajectory[i][0] - ego_x) ** 2 + (trajectory[i][1] - ego_y) ** 2)
        )
        target_idx = min(closest_idx + 5, len(trajectory) - 1)
        target_x, target_y = trajectory[target_idx][:2]

        # 4. Detect obstacles in front and on the trajectory
        min_distance_to_obstacle = float('inf')
        blocking_obstacle = None
        time_horizon = 2.0  # seconds to predict future position
        safe_distance = 10.0  # meters safe following distance

        for obstacle in filtered_obstacles:
            obstacle_location = obstacle.get_location()
            obstacle_velocity = obstacle.get_velocity()
            obstacle_x, obstacle_y = obstacle_location.x, obstacle_location.y
            obstacle_distance = math.sqrt((obstacle_x - ego_x) ** 2 + (obstacle_y - ego_y) ** 2)

            # Check if obstacle is in front
            if is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y):
                # Check if obstacle intersects with the trajectory
                for point in trajectory:
                    traj_x, traj_y = point[:2]
                    if math.sqrt((traj_x - obstacle_x) ** 2 + (traj_y - obstacle_y) ** 2) < 2.0:  # Check within a radius
                        blocking_obstacle = obstacle
                        min_distance_to_obstacle = min(min_distance_to_obstacle, obstacle_distance)
                        break

        # 5. Path planning or direct target handling
        if self.step % 1 == 0 and blocking_obstacle and min_distance_to_obstacle < 25.0:
            # Try A* path planning
            planned_path = self.plan_path_with_a_star((ego_x, ego_y), waypoints[0], filtered_obstacles, boundary)

            if planned_path:
                # Use the first point in the planned path as the new target
                target_x, target_y = planned_path[0][0], planned_path[0][1]
            else:
                # Follow the obstacle in front if no alternative path found
                print("Path Not Found!!!!!")
                target_x, target_y = obstacle_x, obstacle_y

                relative_speed = math.sqrt(obstacle_velocity.x ** 2 + obstacle_velocity.y ** 2) - ego_vel

                if min_distance_to_obstacle < safe_distance:  # If too close, slow down
                    control.throttle = 0.0
                    control.brake = 1.0  # Full brake
                else:
                    # Adjust throttle to maintain safe following distance
                    proportional_gain = 0.1
                    control.throttle = max(0.0, min(1.0, proportional_gain * (safe_distance - min_distance_to_obstacle + relative_speed)))
                    control.brake = 0.0 if relative_speed > 0 else min(1.0, -relative_speed * 0.5)
        else:
            # Speed control based on curvature
            if len(trajectory) > 2:
                curvature = self.compute_curvature(trajectory, closest_idx)
                future_curvature = self.compute_curvature(trajectory, closest_idx + 2) if closest_idx + 2 < len(trajectory) else curvature

                if curvature > np.radians(40):  # Very sharp turn
                    control.throttle = 0.2 if ego_vel <= 3.0 else 0.0
                    control.brake = 1.0
                elif curvature > np.radians(30):
                    control.throttle = 0.3 if ego_vel <= 5.0 else 0.0
                    control.brake = 0.9
                elif curvature > np.radians(20):
                    control.throttle = 0.4 if ego_vel <= 8.0 else 0.0
                    control.brake = 0.8
                elif curvature > np.radians(10):
                    control.throttle = 0.6 if ego_vel <= 12.0 else 0.0
                    control.brake = 0.6
                else:
                    control.throttle = 0.7 if ego_vel <= 20.0 else 0.5
                    control.brake = 0.0
        

        # No blocking obstacles, proceed with normal trajectory
        dx = target_x - ego_x
        dy = target_y - ego_y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_diff = angle_to_target - ego_yaw
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        max_steering_angle = 45.0
        control.steer = max(-1.0, min(1.0, angle_diff / max_steering_angle))


        self.step += 1
        return control


    def compute_local_trajectory(self, boundary):
        """
        Compute a local trajectory from the center line of the given boundary.
        """
        left_boundary = boundary[0]
        right_boundary = boundary[1]
        trajectory = []

        for i in range(len(left_boundary)):
            left_point = np.array([left_boundary[i].transform.location.x, left_boundary[i].transform.location.y])
            right_point = np.array([right_boundary[i].transform.location.x, right_boundary[i].transform.location.y])
            center_point = (left_point + right_point) / 2
            trajectory.append((center_point[0], center_point[1], 0))  # Assuming flat trajectory

        return trajectory

    def compute_curvature(self, trajectory, idx):
        """
        Compute curvature at a given trajectory index.
        """
        if idx < 1 or idx >= len(trajectory) - 1:
            return 0.0

        prev_point = np.array(trajectory[idx - 1][:2])
        curr_point = np.array(trajectory[idx][:2])
        next_point = np.array(trajectory[idx + 1][:2])

        vec1 = curr_point - prev_point
        vec2 = next_point - curr_point
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product > 0:
            return np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        return 0.0


    def compute_local_trajectory(self, boundary):
        """
        Compute a local trajectory from the center line of the given boundary.
        """
        left_boundary = boundary[0]
        right_boundary = boundary[1]
        trajectory = []

        for i in range(len(left_boundary)):
            left_point = np.array([left_boundary[i].transform.location.x, left_boundary[i].transform.location.y])
            right_point = np.array([right_boundary[i].transform.location.x, right_boundary[i].transform.location.y])
            center_point = (left_point + right_point) / 2
            trajectory.append((center_point[0], center_point[1], 0))  # Assuming flat trajectory

        return trajectory


    def visualize_path(self, boundary, obstacles, start_pos, goal_pos, path):
        """
        Visualize the boundaries, obstacles with yaw, start/goal positions, and the planned path.

        Args:
            boundary: [left_boundary, right_boundary] waypoints.
            obstacles: List of obstacles.
            start_pos: (x, y) of the start position (agent as a point).
            goal_pos: (x, y) of the goal position.
            path: List of (x, y) points in the planned path.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot boundaries
        left_boundary = boundary[0]
        right_boundary = boundary[1]
        left_x = [wp.transform.location.x for wp in left_boundary]
        left_y = [wp.transform.location.y for wp in left_boundary]
        right_x = [wp.transform.location.x for wp in right_boundary]
        right_y = [wp.transform.location.y for wp in right_boundary]

        ax.plot(left_x, left_y, 'k--', label='Left Boundary')
        ax.plot(right_x, right_y, 'k--', label='Right Boundary')

        # Plot obstacles as rotated rectangles
        for obstacle in obstacles:
            obs_loc = obstacle.get_location()
            obs_x, obs_y = obs_loc.x, obs_loc.y
            obs_yaw = obstacle.get_transform().rotation.yaw  # Get obstacle yaw
            obs_length = 5.0  # Example length
            obs_width = 1.6   # Example width

            # Compute the four corners of the rotated rectangle
            half_length = obs_length / 2
            half_width = obs_width / 2
            corners = [
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width)
            ]

            # Rotate corners based on yaw
            cos_yaw = math.cos(math.radians(obs_yaw))
            sin_yaw = math.sin(math.radians(obs_yaw))
            rotated_corners = [
                (cos_yaw * cx - sin_yaw * cy + obs_x, sin_yaw * cx + cos_yaw * cy + obs_y)
                for cx, cy in corners
            ]

            # Create a polygon for the rotated rectangle
            polygon = plt.Polygon(rotated_corners, color='red', alpha=0.5,
                                label='Obstacle(Over Estimated)' if not any(['Obstacle' in text.get_text() for text in ax.texts]) else None)
            ax.add_patch(polygon)

        # Plot start and goal positions
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start Position')
        ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal Position')

        # Plot the planned path
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Planned Path')

        # Add legend and grid
        ax.legend()
        ax.grid(True)
        ax.set_title("A* Path Visualization (Agent as Point)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        plt.show()



    def compute_curvature(self, trajectory, idx):
        """
        Compute curvature at a given trajectory index.
        """
        if idx < 1 or idx >= len(trajectory) - 1:
            return 0.0

        prev_point = np.array(trajectory[idx - 1][:2])
        curr_point = np.array(trajectory[idx][:2])
        next_point = np.array(trajectory[idx + 1][:2])

        vec1 = curr_point - prev_point
        vec2 = next_point - curr_point
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product > 0:
            return np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        return 0.0

    def plan_path_with_a_star(self, start_pos, goal_pos, obstacles, boundary):
        """
        Plan a path from start_pos to goal_pos using A* within given boundaries, avoiding obstacles.
        Neighbors are limited to positions in front of the car.
        """
        resolution = 0.25  # Finer resolution (meters per grid cell)
        margin = 5.0       # Extend search area by 5 meters beyond boundaries

        left_boundary = boundary[0]
        right_boundary = boundary[1]

        vehicle_length = 5.0  # Meters
        vehicle_width = 2.0   # Meters
        safety_margin = 0.5   # Additional clearance

        # Compute extended boundary extents
        min_x = min(pt.transform.location.x for pt in left_boundary + right_boundary) - margin
        max_x = max(pt.transform.location.x for pt in left_boundary + right_boundary) + margin
        min_y = min(pt.transform.location.y for pt in left_boundary + right_boundary) - margin
        max_y = max(pt.transform.location.y for pt in left_boundary + right_boundary) + margin

        def to_grid(x, y):
            """Convert world coordinates to grid coordinates."""
            gx = int((x - min_x) / resolution)
            gy = int((y - min_y) / resolution)
            return gx, gy

        def to_world(gx, gy):
            """Convert grid coordinates to world coordinates."""
            wx = gx * resolution + min_x
            wy = gy * resolution + min_y
            return wx, wy

        start_g = to_grid(*start_pos)
        goal_g = to_grid(goal_pos[0], goal_pos[1])

        def is_free(x, y):
            """
            Check if a position (x, y) is free of obstacles and if the vehicle can fit.
            """
            if x < min_x or x > max_x or y < min_y or y > max_y:
                return False

            for obs in obstacles:
                obs_loc = obs.get_location()
                obs_x, obs_y = obs_loc.x, obs_loc.y
                obs_yaw = obs.get_transform().rotation.yaw
                obs_length = vehicle_length
                obs_width = vehicle_width

                half_length = obs_length / 2 + safety_margin
                half_width = obs_width / 2 + safety_margin

                corners = [
                    [-half_length, -half_width],
                    [half_length, -half_width],
                    [half_length, half_width],
                    [-half_length, half_width]
                ]

                cos_yaw = math.cos(math.radians(obs_yaw))
                sin_yaw = math.sin(math.radians(obs_yaw))
                rotated_corners = [
                    (cos_yaw * cx - sin_yaw * cy + obs_x, sin_yaw * cx + cos_yaw * cy + obs_y)
                    for cx, cy in corners
                ]

                if is_point_in_polygon((x, y), rotated_corners):
                    return False

            return True

        def is_point_in_polygon(point, polygon):
            px, py = point
            inside = False
            for i in range(len(polygon)):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % len(polygon)]
                if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        def get_front_neighbors(current, heading_angle):
            """
            Get neighbors limited to the forward direction based on the car's heading angle.
            """
            forward_neighbors = []
            angles = [-30, -15, 0, 15, 30]  # Angular offsets in degrees
            step_size = 1  # Step size in grid units

            for angle in angles:
                rad = math.radians(heading_angle + angle)
                dx = int(step_size * math.cos(rad))
                dy = int(step_size * math.sin(rad))
                forward_neighbors.append((current[0] + dx, current[1] + dy))

            return forward_neighbors

        # Initialize A* search
        open_set = []
        heappush(open_set, (0, start_g))
        came_from = {}
        g_score = {start_g: 0}
        h = lambda n: math.sqrt((n[0] - goal_g[0]) ** 2 + (n[1] - goal_g[1]) ** 2)

        visited = set()

        while open_set:
            _, current = heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal_g:
                path = []
                node = current
                while node in came_from:
                    wx, wy = to_world(node[0], node[1])
                    path.append((wx, wy))
                    node = came_from[node]
                path.append(start_pos)
                path.reverse()
                self.visualize_path(boundary, obstacles, start_pos, goal_pos, path)
                return path

            # Assume heading_angle is retrieved from the agent or calculated dynamically
            heading_angle = 0  # Replace this with the actual heading angle of the car in degrees
            neighbors = get_front_neighbors(current, heading_angle)

            for neighbor in neighbors:
                nx, ny = neighbor
                wx, wy = to_world(nx, ny)
                if not is_free(wx, wy):
                    continue

                tentative_g_score = g_score[current] + math.sqrt((nx - current[0]) ** 2 + (ny - current[1]) ** 2)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + h(neighbor)
                    heappush(open_set, (f_score, neighbor))

        print("A* could not find a path.")
        return []



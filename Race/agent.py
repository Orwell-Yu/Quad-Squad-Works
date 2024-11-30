import carla
import time
import numpy as np
import math
import random
import math


class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y, front_angle_threshold=10):
    """
    Check if an obstacle is in front of the ego vehicle within a specified angular range.

    Args:
        ego_x, ego_y: Ego vehicle's position.
        ego_yaw: Ego vehicle's yaw angle (in degrees).
        obstacle_x, obstacle_y: Obstacle position.
        front_angle_threshold: Angular range (in degrees) to consider an obstacle as "in front".

    Returns:
        bool: True if the obstacle is in front, False otherwise.
    """
    # Calculate the ego vehicle's forward vector
    ego_yaw_rad = math.radians(ego_yaw)
    forward_vector = np.array([math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)])

    # Calculate the vector to the obstacle
    obstacle_vector = np.array([obstacle_x - ego_x, obstacle_y - ego_y])
    obstacle_distance = np.linalg.norm(obstacle_vector)

    # Normalize the obstacle vector
    if obstacle_distance > 0:
        obstacle_vector /= obstacle_distance

    # Compute the angle between the forward vector and obstacle vector
    dot_product = np.dot(forward_vector, obstacle_vector)
    angle_to_obstacle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))

    # Check if the angle is within the threshold
    return angle_to_obstacle < front_angle_threshold


class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.step_size = 0.1
        self.max_iterations = 500

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        control = carla.VehicleControl()


        # 1. Get Ego Vehicle Position
        ego_location = transform.location
        ego_x, ego_y = ego_location.x, ego_location.y
        ego_yaw = transform.rotation.yaw
        ego_vel = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        
        # 2. Determine Target Waypoint
        target_waypoint = waypoints[0]
        target_x, target_y = target_waypoint[0], target_waypoint[1]

        # 3. Check for Obstacles and Plan Path Using RRT
        min_distance_to_obstacle = float('inf')
        obstacle_in_front = False
        for obstacle in filtered_obstacles:
            obstacle_location = obstacle.get_location()
            obstacle_x, obstacle_y = obstacle_location.x, obstacle_location.y
            
            obstacle_distance = math.sqrt(
                (obstacle_x - ego_x) ** 2 + (obstacle_y - ego_y) ** 2
            )
            
            # Check if the obstacle is in front
            if is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y):
                obstacle_in_front = True
                min_distance_to_obstacle = min(min_distance_to_obstacle, obstacle_distance)
        
        # If obstacles are detected within a certain range, use RRT to plan a path
        if obstacle_in_front and min_distance_to_obstacle < 10.0:
            # Abstract RRT Planner function to avoid obstacles
            planned_path = self.plan_path_with_rrt(ego_location, waypoints, filtered_obstacles, boundary)
            
            # Use the first point in the planned path as the new target
            if planned_path:
                target_x, target_y = planned_path[0][0], planned_path[0][1]

        # 3. Calculate Distance and Angle to Target Waypoint
        dx = target_x - ego_x
        dy = target_y - ego_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.degrees(math.atan2(dy, dx))
        
        # 4. Calculate Steering Angle (Simple Proportional Controller)
        angle_diff = angle_to_target - ego_yaw
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        # Normalize the steering value to [-1, 1]
        control.steer = max(-1.0, min(1.0, angle_diff / 45.0))

        # 5. Speed Control
        # Reduce speed if an obstacle is within 10 meters
        min_distance_to_obstacle = float('inf')
        obstacle_in_front = False
        for obstacle in filtered_obstacles:
            obstacle_location = obstacle.get_location()
            obstacle_x, obstacle_y = obstacle_location.x, obstacle_location.y
            
            obstacle_distance = math.sqrt(
                (obstacle_x - ego_x) ** 2 + (obstacle_y - ego_y) ** 2
            )
            
            # Check if the obstacle is in front
            if is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y):
                obstacle_in_front = True
                min_distance_to_obstacle = min(min_distance_to_obstacle, obstacle_distance)
        

        if obstacle_in_front:
            if min_distance_to_obstacle < 20.0:
                # 高速时动态计算刹车和油门
                if ego_vel > 30.0:  # 高速
                    if min_distance_to_obstacle < 10.0:
                        control.brake = min(1.0, 1.0 - (min_distance_to_obstacle / 10.0))  # 强制全刹车
                        control.throttle = 0.0
                    else:
                        control.brake = min(0.7, 0.7 - (min_distance_to_obstacle / 20.0))  # 中等刹车
                        control.throttle = max(0.2, 0.5 - (20.0 - min_distance_to_obstacle) / 20.0)
                else:  # 中低速
                    if min_distance_to_obstacle < 10.0:
                        control.brake = min(1.0, 1.0 - (min_distance_to_obstacle / 10.0))
                        control.throttle = 0.0
                    else:
                        control.brake = min(0.5, 0.5 - (min_distance_to_obstacle / 20.0))
                        control.throttle = max(0.3, 0.6 - (20.0 - min_distance_to_obstacle) / 20.0)
            else:
                # 障碍物在20米以外，限制速度但不刹车
                control.brake = 0.0
                control.throttle = max(0.4, 0.7 - ego_vel / 50.0)  # 限制速度增长
        else:
            if len(waypoints) > 2:
                wp1 = np.array([waypoints[0][0], waypoints[0][1]])
                wp2 = np.array([waypoints[1][0], waypoints[1][1]])
                wp3 = np.array([waypoints[2][0], waypoints[2][1]])

                # Compute vectors
                vec1 = wp2 - wp1
                vec2 = wp3 - wp2

                # Calculate the angle between the two vectors
                dot_product = np.dot(vec1, vec2)
                norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

                # Avoid division by zero
                if norm_product > 0:
                    curvature = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
                else:
                    curvature = 0.0
            else:
                curvature = 0.0

            # Set thresholds for curvature
            curvature_brake_threshold = np.radians(30)  # Braking for very sharp turns (~30 degrees or more)
            curvature_throttle_threshold = np.radians(10)  # Reduce throttle for moderate turns (~10 degrees or more)

            # Determine throttle and braking based on curvature
            if curvature > curvature_brake_threshold:
                # Excessive curvature, apply brakes
                if ego_vel > 5.0:
                    control.throttle = 0.1
                    control.brake = 0.9  # Full brake
                else:
                    control.throttle = 0.6
                    control.brake = 0.0
            elif curvature > curvature_throttle_threshold:
                # Moderate curvature, reduce throttle
                throttle_value = max(0.5, 1.0 - curvature * 2.0)  # Adjust sensitivity with the multiplier
                control.throttle = throttle_value
                control.brake = 0.0
            else:
                if ego_vel >= 35:
                    control.throttle = 0.5
                    control.brake = 0.0
                else:
                    if ego_vel < 20:
                        control.throttle = 0.85
                        control.brake = 0.0
                    else:
                        control.throttle = 0.75
                        control.brake = 0.0

        
        # 6. Boundary Check (Stay within the track boundaries)
        left_boundary = boundary[0]
        right_boundary = boundary[1]
        left_boundary_x = left_boundary[0].transform.location.x
        left_boundary_y = left_boundary[0].transform.location.y
        right_boundary_x = right_boundary[0].transform.location.x
        right_boundary_y = right_boundary[0].transform.location.y

        # Check distance to left and right boundaries
        left_dist = math.sqrt((ego_x - left_boundary_x) ** 2 + (ego_y - left_boundary_y) ** 2)
        right_dist = math.sqrt((ego_x - right_boundary_x) ** 2 + (ego_y - right_boundary_y) ** 2)

        # If too close to a boundary, adjust steering
        if left_dist < 0.8:
            control.steer += 0.1  # Steer right
        elif right_dist < 0.8:
            control.steer -= 0.1  # Steer left

        # Return the control commands
        return control



    def plan_path_with_rrt(self, start_location, waypoints, obstacles, boundary):
        """
        Plan a collision-free path using RRT.

        Args:
        start_location: carla.Location
        waypoints: List[[x, y, z], ...]
        obstacles: List[carla.Actor()]
        boundary: List[List[left_boundary], List[right_boundary]]

        Return: List[[x, y], ...] - A list of points defining the planned path.
        """
        start_node = RRTNode(start_location.x, start_location.y)
        goal_node = RRTNode(waypoints[0][0], waypoints[0][1])
        tree = [start_node]

        for _ in range(self.max_iterations):
            # Step 1: Sample a random point within the boundary
            rand_point = self.sample_random_point(boundary)

            # Step 2: Find the nearest node in the tree
            nearest_node = self.get_nearest_node(tree, rand_point)

            # Step 3: Steer towards the sampled point
            new_node = self.steer(nearest_node, rand_point)

            # Step 4: Check for collisions
            if self.is_collision_free(nearest_node, new_node, obstacles, boundary):
                tree.append(new_node)

                # Step 5: Check if the goal is reached
                if self.distance(new_node, goal_node) < self.step_size:
                    return self.extract_path(new_node)

        # If no path is found, return an empty path
        return []

    def sample_random_point(self, boundary):
        """Sample a random point within the given boundaries."""
        left_boundary = boundary[0]
        right_boundary = boundary[1]

        # Randomly select a point between left and right boundaries
        min_x = min(left_boundary[0].transform.location.x, right_boundary[0].transform.location.x)
        max_x = max(left_boundary[-1].transform.location.x, right_boundary[-1].transform.location.x)
        min_y = min(left_boundary[0].transform.location.y, right_boundary[0].transform.location.y)
        max_y = max(left_boundary[-1].transform.location.y, right_boundary[-1].transform.location.y)

        rand_x = random.uniform(min_x, max_x)
        rand_y = random.uniform(min_y, max_y)
        return (rand_x, rand_y)

    def get_nearest_node(self, tree, point):
        """Find the nearest node in the tree to the given point."""
        nearest_node = tree[0]
        min_distance = self.distance(nearest_node, RRTNode(point[0], point[1]))

        for node in tree:
            dist = self.distance(node, RRTNode(point[0], point[1]))
            if dist < min_distance:
                nearest_node = node
                min_distance = dist

        return nearest_node

    def steer(self, from_node, to_point):
        """Extend the tree towards the sampled point."""
        angle = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        new_x = from_node.x + self.step_size * math.cos(angle)
        new_y = from_node.y + self.step_size * math.sin(angle)
        return RRTNode(new_x, new_y, parent=from_node)

    def is_collision_free(self, from_node, to_node, obstacles, boundary):
        """
        Check if the path between two nodes is collision-free.

        Args:
            from_node (RRTNode): Starting node of the path.
            to_node (RRTNode): Ending node of the path.
            obstacles (list): List of obstacle actors.
            boundary (list): [left_boundary, right_boundary], each contains waypoints.

        Returns:
            bool: True if the path is collision-free, False otherwise.
        """
        # Check for collision with obstacles
        for obstacle in obstacles:
            obs_location = obstacle.get_location()
            obs_x, obs_y = obs_location.x, obs_location.y

            # Calculate the distance from the obstacle to the path
            obstacle_distance = self.distance_to_line(from_node, to_node, (obs_x, obs_y))

            # Use obstacle radius if available, otherwise default to 2.0
            obstacle_radius = getattr(obstacle, 'bounding_box', None)
            if obstacle_radius:
                collision_radius = obstacle_radius.extent.x  # Adjust based on actual bounding box
            else:
                collision_radius = 3.5

            # If the obstacle is too close to the path, return False
            if obstacle_distance < collision_radius:
                print(f"Collision detected with obstacle at ({obs_x}, {obs_y})")
                return False

        # # Check if the path is within the track boundaries
        # left_boundary = boundary[0]
        # right_boundary = boundary[1]

        # # Check both from_node and to_node
        # if not self.is_within_boundaries(from_node, left_boundary, right_boundary) or \
        # not self.is_within_boundaries(to_node, left_boundary, right_boundary):
        #     print(f"Node ({to_node.x}, {to_node.y}) is out of track boundaries.")
        #     return False

        # Path is collision-free and within boundaries
        return True

    def is_within_boundaries(self, node, left_boundary, right_boundary):
        """
        Check if a node is within the boundaries defined by the track.

        Args:
            node (RRTNode): Node to check.
            left_boundary (list): List of left boundary waypoints.
            right_boundary (list): List of right boundary waypoints.

        Returns:
            bool: True if the node is within boundaries, False otherwise.
        """
        node_x, node_y = node.x, node.y

        # Ensure the node is within the x-range of the boundaries
        min_x = min(pt.transform.location.x for pt in left_boundary)
        max_x = max(pt.transform.location.x for pt in right_boundary)
        if not (min_x <= node_x <= max_x):
            return False

        # Ensure the node is within the y-range defined by the boundaries
        min_y = min(pt.transform.location.y for pt in left_boundary)
        max_y = max(pt.transform.location.y for pt in right_boundary)
        if not (min_y <= node_y <= max_y):
            return False

        return True

    def extract_path(self, node):
        """Extract the path from the RRT tree by backtracking from the goal."""
        path = []
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.reverse()
        return path

    def distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes."""
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def distance_to_line(self, from_node, to_node, point):
        """Calculate the perpendicular distance from a point to a line segment."""
        x1, y1 = from_node.x, from_node.y
        x2, y2 = to_node.x, to_node.y
        px, py = point

        norm = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)) / norm
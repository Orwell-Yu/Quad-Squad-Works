import carla
import time
import numpy as np
import math
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


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
        self.step_size = 0.5
        self.max_iterations = 2000
        self.velocity_history = []  # Store (time, velocity)

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



    # def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
    #     """
    #     Execute one step of navigation.

    #     Args:
    #     filtered_obstacles
    #         - Type:        List[carla.Actor(), ...]
    #         - Description: All actors except for EGO within sensoring distance
    #     waypoints 
    #         - Type:         List[[x,y,z], ...] 
    #         - Description:  List All future waypoints to reach in (x,y,z) format
    #     vel
    #         - Type:         carla.Vector3D 
    #         - Description:  Ego's current velocity in (x, y, z) in m/s
    #     transform
    #         - Type:         carla.Transform 
    #         - Description:  Ego's current transform
    #     boundary 
    #         - Type:         List[List[left_boundry], List[right_boundry]]
    #         - Description:  left/right boundary each consists of 20 waypoints,
    #                         they defines the track boundary of the next 20 meters.

    #     Return: carla.VehicleControl()
    #     """
    #     control = carla.VehicleControl()


    #     # 1. Get Ego Vehicle Position
    #     ego_location = transform.location
    #     ego_x, ego_y = ego_location.x, ego_location.y
    #     ego_yaw = transform.rotation.yaw
    #     ego_vel = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        


    #     # 2. Determine Target Waypoint
    #     target_waypoint = waypoints[0]
    #     target_x, target_y = target_waypoint[0], target_waypoint[1]

    #     # 3. Check for Obstacles and Plan Path Using RRT
    #     min_distance_to_obstacle = float('inf')
    #     obstacle_in_front = False
    #     for obstacle in filtered_obstacles:
    #         obstacle_location = obstacle.get_location()
    #         obstacle_x, obstacle_y = obstacle_location.x, obstacle_location.y
            
    #         obstacle_distance = math.sqrt(
    #             (obstacle_x - ego_x) ** 2 + (obstacle_y - ego_y) ** 2
    #         )
            
    #         # Check if the obstacle is in front
    #         if is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y):
    #             obstacle_in_front = True
    #             min_distance_to_obstacle = min(min_distance_to_obstacle, obstacle_distance)
        
    #     # If obstacles are detected within a certain range, use RRT to plan a path
    #     if obstacle_in_front and min_distance_to_obstacle < 10.0:
    #         # Abstract RRT Planner function to avoid obstacles
    #         planned_path = self.plan_path_with_rrt(ego_location, waypoints, filtered_obstacles, boundary)
            
    #         # Use the first point in the planned path as the new target
    #         if planned_path:
    #             target_x, target_y = planned_path[0][0], planned_path[0][1]

    #     # 3. Calculate Distance and Angle to Target Waypoint
    #     dx = target_x - ego_x
    #     dy = target_y - ego_y
    #     distance_to_target = math.sqrt(dx**2 + dy**2)
    #     angle_to_target = math.degrees(math.atan2(dy, dx))
        
    #     # 4. Calculate Steering Angle (Simple Proportional Controller)
    #     angle_diff = angle_to_target - ego_yaw
    #     while angle_diff > 180:
    #         angle_diff -= 360
    #     while angle_diff < -180:
    #         angle_diff += 360
        
    #     # Normalize the steering value to [-1, 1]
    #     max_steering_angle = 45.0
    #     control.steer = max(-1.0, min(1.0, angle_diff / max_steering_angle))

    #     # 5. Speed Control
    #     # Reduce speed if an obstacle is within 10 meters
    #     min_distance_to_obstacle = float('inf')
    #     obstacle_in_front = False
    #     for obstacle in filtered_obstacles:
    #         obstacle_location = obstacle.get_location()
    #         obstacle_x, obstacle_y = obstacle_location.x, obstacle_location.y
            
    #         obstacle_distance = math.sqrt(
    #             (obstacle_x - ego_x) ** 2 + (obstacle_y - ego_y) ** 2
    #         )
            
    #         # Check if the obstacle is in front
    #         if is_obstacle_in_front(ego_x, ego_y, ego_yaw, obstacle_x, obstacle_y):
    #             obstacle_in_front = True
    #             min_distance_to_obstacle = min(min_distance_to_obstacle, obstacle_distance)
        

    #     if obstacle_in_front:
    #         if min_distance_to_obstacle < 20.0:
    #             # 高速时动态计算刹车和油门
    #             if ego_vel > 30.0:  # 高速
    #                 if min_distance_to_obstacle < 10.0:
    #                     control.brake = min(1.0, 1.0 - (min_distance_to_obstacle / 10.0))  # 强制全刹车
    #                     control.throttle = 0.0
    #                 else:
    #                     control.brake = min(0.7, 0.7 - (min_distance_to_obstacle / 20.0))  # 中等刹车
    #                     control.throttle = max(0.2, 0.5 - (20.0 - min_distance_to_obstacle) / 20.0)
    #             else:  # 中低速
    #                 if min_distance_to_obstacle < 10.0:
    #                     control.brake = min(1.0, 1.0 - (min_distance_to_obstacle / 10.0))
    #                     control.throttle = 0.0
    #                 else:
    #                     control.brake = min(0.5, 0.5 - (min_distance_to_obstacle / 20.0))
    #                     control.throttle = max(0.3, 0.6 - (20.0 - min_distance_to_obstacle) / 20.0)
    #         else:
    #             # 障碍物在20米以外，限制速度但不刹车
    #             control.brake = 0.0
    #             control.throttle = max(0.4, 0.7 - ego_vel / 50.0)  # 限制速度增长
    #     else:
    #         # if len(waypoints) > 2:
    #         #     wp1 = np.array([waypoints[0][0], waypoints[0][1]])
    #         #     wp2 = np.array([waypoints[1][0], waypoints[1][1]])
    #         #     wp3 = np.array([waypoints[2][0], waypoints[2][1]])

    #         #     # Compute vectors
    #         #     vec1 = wp2 - wp1
    #         #     vec2 = wp3 - wp2

    #         #     # Calculate the angle between the two vectors
    #         #     dot_product = np.dot(vec1, vec2)
    #         #     norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    #         #     # Avoid division by zero
    #         #     if norm_product > 0:
    #         #         curvature = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    #         #         # print(curvature)
    #         #     else:
    #         #         curvature = 0.0
    #         # else:
    #         #     curvature = 0.0

    #         # Initialize curvature values
    #         # curvature = 0.0
    #         # future_curvature = 0.0  # Add this to avoid UnboundLocalError
    #         # # cur_pos = self.vehicle.get_location()
    #         # # print(cur_pos)
    #         # pos=(ego_x,ego_y)
    #         # print(pos)

    #         # if len(waypoints) > 5:
    #         #     wp1 = np.array([waypoints[0][0], waypoints[0][1]])
    #         #     wp2 = np.array([waypoints[1][0], waypoints[1][1]])
    #         #     wp3 = np.array([waypoints[2][0], waypoints[2][1]])

    #         #     # Compute vectors
    #         #     vec1 = wp1 - pos
    #         #     vec2 = wp2 - wp1

    #         #     # Calculate the angle between the two vectors
    #         #     dot_product = np.dot(vec1, vec2)
    #         #     norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    #         #     # Avoid division by zero
    #         #     if norm_product > 0:
    #         #         curvature = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    #         #     else:
    #         #         curvature = 0.0

    #         #     # Calculate future curvature using waypoints 3, 4, 5
    #         #     wp4 = np.array([waypoints[3][0], waypoints[3][1]])
    #         #     wp5 = np.array([waypoints[4][0], waypoints[4][1]])
    #         #     wp6 = np.array([waypoints[5][0], waypoints[5][1]])

    #         #     vec3 = wp4 - wp3
    #         #     vec4 = wp5 - wp4

    #         #     future_dot_product = np.dot(vec3, vec4)
    #         #     future_norm_product = np.linalg.norm(vec3) * np.linalg.norm(vec4)

    #         #     if future_norm_product > 0:
    #         #         future_curvature = np.arccos(np.clip(future_dot_product / future_norm_product, -1.0, 1.0))
    #         #     else:
    #         #         future_curvature = 0.0

    #         left_boundary = boundary[0]
    #         right_boundary = boundary[1]
    #         # print(left_boundary)
    #         # print(right_boundary)

    #         if len(left_boundary) >= 3 and len(right_boundary) >= 3:
    #             # Take three consecutive points from the left boundary for curvature approximation
    #             left_p1 = np.array([left_boundary[0].transform.location.x, left_boundary[0].transform.location.y])
    #             left_p2 = np.array([left_boundary[5].transform.location.x, left_boundary[5].transform.location.y])
    #             left_p3 = np.array([left_boundary[10].transform.location.x, left_boundary[10].transform.location.y])

    #             # Take three consecutive points from the right boundary for curvature approximation
    #             right_p1 = np.array([right_boundary[0].transform.location.x, right_boundary[0].transform.location.y])
    #             right_p2 = np.array([right_boundary[5].transform.location.x, right_boundary[5].transform.location.y])
    #             right_p3 = np.array([right_boundary[10].transform.location.x, right_boundary[10].transform.location.y])

    #             # Compute vectors and curvature for the centerline
    #             center_p1 = (left_p1 + right_p1) / 2
    #             center_p2 = (left_p2 + right_p2) / 2
    #             center_p3 = (left_p3 + right_p3) / 2

    #             vec1 = center_p2 - center_p1
    #             vec2 = center_p3 - center_p2

    #             dot_product = np.dot(vec1, vec2)
    #             norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    #             # Avoid division by zero
    #             curvature = 0.0
    #             if norm_product > 0:
    #                 curvature = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))

    #         else:
    #             curvature = 0.0  # Default curvature if boundary data is insufficient

    #         if len(left_boundary) >= 6 and len(right_boundary) >= 6:

    #             # Compute future curvature using the next three boundary points
    #             left_p4 = np.array([left_boundary[35].transform.location.x, left_boundary[35].transform.location.y])
    #             left_p5 = np.array([left_boundary[40].transform.location.x, left_boundary[40].transform.location.y])
    #             left_p6 = np.array([left_boundary[45].transform.location.x, left_boundary[45].transform.location.y])

    #             right_p4 = np.array([right_boundary[35].transform.location.x, right_boundary[35].transform.location.y])
    #             right_p5 = np.array([right_boundary[40].transform.location.x, right_boundary[40].transform.location.y])
    #             right_p6 = np.array([right_boundary[45].transform.location.x, right_boundary[45].transform.location.y])

    #             center_p4 = (left_p4 + right_p4) / 2
    #             center_p5 = (left_p5 + right_p5) / 2
    #             center_p6 = (left_p6 + right_p6) / 2

    #             vec3 = center_p5 - center_p4
    #             vec4 = center_p6 - center_p5

    #             future_dot_product = np.dot(vec3, vec4)
    #             future_norm_product = np.linalg.norm(vec3) * np.linalg.norm(vec4)

    #             future_curvature = 0.0
    #             if future_norm_product > 0:
    #                 future_curvature = np.arccos(np.clip(future_dot_product / future_norm_product, -1.0, 1.0))
    #         else:
    #             future_curvature = 0.0

    #         # Set thresholds for curvature
    #         curvature_brake_threshold = np.radians(30)  # Braking for very sharp turns (~30 degrees or more)
    #         curvature_throttle_threshold = np.radians(10)  # Reduce throttle for moderate turns (~10 degrees or more)

    #         print(curvature)
    #         print(future_curvature)
    #         print("----------------------")

    #         # Determine throttle and braking based on curvature
    #         # if curvature > curvature_brake_threshold:
    #             # Excessive curvature, apply brakes
    #         #     if ego_vel > 7:
    #         #         control.throttle = 0.0
    #         #         control.brake = 0.8  # Full brake
    #         #     else:
    #         #         control.throttle = 0.9
    #         #         control.brake = 0.0
    #         # elif curvature > curvature_throttle_threshold:
    #         #     # Moderate curvature, reduce throttle
    #         #     throttle_value = max(0.5, 1.0 - curvature * 2.0)  # Adjust sensitivity with the multiplier
    #         #     control.throttle = throttle_value
    #         #     control.brake = 0.0
    #         # else:
    #         #     if ego_vel >= 35:
    #         #         control.throttle = 0.75
    #         #         control.brake = 0.0
    #         #     else:
    #         #         if ego_vel < 20:
    #         #             control.throttle = 0.9
    #         #             control.brake = 0.0
    #         #         else:
    #         #             control.throttle = 0.8
    #         #             control.brake = 0.0

    #         # if ego_vel < 15:
    #         #     control.throttle = 0.3
    #         #     control.brake = 0.0
    #         # else:
    #         #     control.throttle = 0.0
    #         #     control.brake = 0.3

    #         # 11/30 modified more fine grainded threshold
            
    #         if curvature > 0.3 and future_curvature > curvature:
    #            if ego_vel > 10.0:
    #                control.throttle = 0.0
    #                control.brake = 0.95  # Full brake
    #             #    print("0")
    #            else:
    #                control.throttle = 0.3
    #                control.brake = 0.0
    #             #    print("1")
        
    #         elif curvature > 0.2 and future_curvature > curvature:
    #             if ego_vel > 15.0:
    #                control.throttle = 0.2
    #                control.brake = 0.8  # Full brake
    #             #    print("2")
    #             else:
    #                control.throttle = 0.5
    #                control.brake = 0.0
    #             #    print("3")

    #         elif curvature > 0.1:
    #             if ego_vel > 30.0:
    #                control.throttle = 0.3
    #                control.brake = 0.8  # Full brake
    #             #    print("4")
    #             else:
    #                control.throttle = 0.65
    #                control.brake = 0.0
    #             #    print("5")

    #         elif curvature > 0.08:
    #            if ego_vel > 50.0:
    #                control.throttle = 0.3
    #                control.brake = 0.75  # Full brake
    #             #    print("6")
    #            else:
    #                control.throttle = 0.8
    #                control.brake = 0.0
    #             #    print("7")

    #         # elif future_curvature > 0.08:
    #         #    if ego_vel > 70.0:
    #         #        control.throttle = 0.1
    #         #        control.brake = 0.7
    #         #    else:
    #         #        control.throttle = 0.7
    #         #        control.brake = 0.0

    #         else:
    #         #    if len(waypoints) > 5:
    #             if ego_vel >= 90:
    #                 control.throttle = 0.5
    #                 control.brake = 0.0
    #                 # print("8")
    #             else:
    #                 if ego_vel < 20:
    #                     control.throttle = 0.75
    #                     control.brake = 0.0
    #                     # print("9")
    #                     # print(curvature," ", wp1," ", wp2," ", wp3)
    #                     # print(future_curvature," ", wp4," ", wp5," ", wp6)
    #                     # print("-------------")
    #                 else:
    #                     control.throttle = 0.75
    #                     control.brake = 0.0
    #                     # print("10")
    #                     # print(curvature," ", wp1," ", wp2," ", wp3)
    #                     # print(future_curvature," ", wp4," ", wp5," ", wp6)
    #                     # print("-------------")


        
    #     # 6. Boundary Check (Stay within the track boundaries)
    #     left_boundary = boundary[0]
    #     right_boundary = boundary[1]
    #     left_boundary_x = left_boundary[0].transform.location.x
    #     left_boundary_y = left_boundary[0].transform.location.y
    #     right_boundary_x = right_boundary[0].transform.location.x
    #     right_boundary_y = right_boundary[0].transform.location.y

    #     # Check distance to left and right boundaries
    #     left_dist = math.sqrt((ego_x - left_boundary_x) ** 2 + (ego_y - left_boundary_y) ** 2)
    #     right_dist = math.sqrt((ego_x - right_boundary_x) ** 2 + (ego_y - right_boundary_y) ** 2)

    #     # If too close to a boundary, adjust steering
    #     if left_dist < 0.8:
    #         control.steer += 0.1  # Steer right
    #     elif right_dist < 0.8:
    #         control.steer -= 0.1  # Steer left

    #     # Return the control commands
    #     return control
    


    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Dynamically calculate the trajectory and control vehicle movement based on updated boundary data in each step.
        """
        control = carla.VehicleControl()

        # 1. Get the current position and state of the Ego vehicle
        ego_location = transform.location
        ego_x, ego_y = ego_location.x, ego_location.y
        ego_yaw = transform.rotation.yaw
        ego_vel = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        current_time = len(self.velocity_history) * self.step_size /2
        self.log_velocity(current_time, ego_vel)

        # 2. Calculate the trajectory based on the boundary
        trajectory = self.compute_local_trajectory(boundary)

        # 3. Find the trajectory point closest to the Ego vehicle as the target point
        closest_idx = min(
            range(len(trajectory)),
            key=lambda i: math.sqrt((trajectory[i][0] - ego_x) ** 2 + (trajectory[i][1] - ego_y) ** 2)
        )
        # Select a "lookahead" point in front of the Ego vehicle
        target_idx = min(closest_idx + 5, len(trajectory) - 1)
        target_x, target_y = trajectory[target_idx][:2]





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





        # 4. Calculate the steering angle adjustment
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

        # 5. Calculate speed control
        if len(trajectory) > 2:
            curvature = self.compute_curvature(trajectory, closest_idx)
            future_curvature = self.compute_curvature(trajectory, closest_idx + 2) if closest_idx + 2 < len(trajectory) else curvature

            # Adjust speed based on curvature
            if curvature > np.radians(40) or future_curvature > np.radians(40):  # Very sharp turn
                if ego_vel > 3.0:  # High speed
                    control.throttle = 0.0
                    control.brake = 1.0  # Full brake
                else:  # Low speed
                    control.throttle = 0.2
                    control.brake = 0.0  # Maintain low speed
            elif curvature > np.radians(30) or future_curvature > np.radians(30):  # Sharp turn
                if ego_vel > 5.0:  # High speed
                    control.throttle = 0.0
                    control.brake = 0.9  # Strong braking
                else:
                    control.throttle = 0.3
                    control.brake = 0.0  # Controlled speed
            elif curvature > np.radians(20) or future_curvature > np.radians(20):  # Moderate turn
                if ego_vel > 8.0:  # High speed
                    control.throttle = 0.0
                    control.brake = 0.8  # Slow down
                else:
                    control.throttle = 0.4
                    control.brake = 0.0  # Smooth drive
            elif curvature > np.radians(10) or future_curvature > np.radians(10):  # Gentle turn
                if ego_vel > 12.0:  # High speed
                    control.throttle = 0.0
                    control.brake = 0.6  # Light braking
                else:
                    control.throttle = 0.6
                    control.brake = 0.4  # Slight deceleration
            else:  # Straight road or mild curvature
                if ego_vel > 20.0:  # Limit acceleration at high speed
                    control.throttle = 0.7
                    control.brake = 0.0
                elif ego_vel < 10.0:  # Accelerate quickly at low speed
                    control.throttle = 0.8
                    control.brake = 0.0
                else:  # Maintain speed at medium speed
                    control.throttle = 0.6
                    control.brake = 0.0

            # 6. Return control commands
        return control






    def compute_local_trajectory(self, boundary):
        """
        use boundary to calculate current 20m(?) trajectory
        """
        left_boundary = boundary[0]
        right_boundary = boundary[1]
        # print(len(left_boundary))
        trajectory = []

        for i in range(len(left_boundary)):
            left_point = np.array([left_boundary[i].transform.location.x, left_boundary[i].transform.location.y])
            right_point = np.array([right_boundary[i].transform.location.x, right_boundary[i].transform.location.y])

            # calc center line
            center_point = (left_point + right_point) / 2
            trajectory.append((center_point[0], center_point[1], 0))  # 假设轨迹是平面的

        return trajectory

    def compute_curvature(self, trajectory, idx):
        """
        use trajectory points to calculate curvature
        """
        if idx < 1 or idx >= len(trajectory) - 1:
            return 0.0  # edge case

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



    def plan_path_with_rrt(self, start_location, waypoints, obstacles, boundary):
        start_node = RRTNode(start_location.x, start_location.y)
        goal_node = RRTNode(waypoints[0][0], waypoints[0][1])
        tree = [start_node]


        fig, ax = plt.subplots()
        plt.ion()

        self.plot_boundaries(ax, boundary)
        self.plot_obstacles(ax, obstacles)

        ax.plot(start_node.x, start_node.y, 'go', markersize=10, label='start')  # 绿色起点
        ax.plot(goal_node.x, goal_node.y, 'ro', markersize=10, label='end')    # 红色目标点

        plt.legend()

        for _ in range(self.max_iterations):
            rand_point = self.sample_random_point(boundary)

            nearest_node = self.get_nearest_node(tree, rand_point)

            new_node = self.steer(nearest_node, rand_point)

            if self.is_collision_free(nearest_node, new_node, obstacles, boundary):
                tree.append(new_node)

                ax.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], '-b')
                plt.pause(0.01)

                if self.distance(new_node, goal_node) < 1:
                    path = self.extract_path(new_node)
                    self.plot_path(ax, path)
                    plt.ioff()
                    plt.show()
                    return path

        plt.ioff()
        plt.show()
        return []
    
    def plot_boundaries(self, ax, boundary):
        left_boundary = boundary[0]
        right_boundary = boundary[1]

        # 提取x和y坐标
        left_x = [wp.transform.location.x for wp in left_boundary]
        left_y = [wp.transform.location.y for wp in left_boundary]
        right_x = [wp.transform.location.x for wp in right_boundary]
        right_y = [wp.transform.location.y for wp in right_boundary]

        # 绘制边界线
        ax.plot(left_x, left_y, 'k--', label='left_boundary')  # 黑色虚线
        ax.plot(right_x, right_y, 'k--', label='right_boundary')
    
    def plot_obstacles(self, ax, obstacles):
        for obstacle in obstacles:
            vx, vy = self.get_obstacle_vertices(obstacle)
            ax.plot(vx, vy, marker='o', linestyle='-', linewidth=2)
            # ax.scatter(vx, vy, color='yellow', label='Obstacle' if 'Obstacle' not in [t.get_text() for t in ax.texts] else "")
        
    def get_obstacle_vertices(self, obstacle):
        bounding_box = obstacle.bounding_box

        # Get the vehicle's transformation in the world
        vehicle_transform = obstacle.get_transform()

        # Get the local vertices of the bounding box
        extent = bounding_box.extent  # carla.Vector3D, half the size of the box in each direction

        # Define the eight vertices of the bounding box in local coordinates
        local_vertices = [
            carla.Location(x=extent.x, y=extent.y, z=extent.z),
            carla.Location(x=extent.x, y=-extent.y, z=extent.z),
            carla.Location(x=-extent.x, y=extent.y, z=extent.z),
            carla.Location(x=-extent.x, y=-extent.y, z=extent.z),
            carla.Location(x=extent.x, y=extent.y, z=-extent.z),
            carla.Location(x=extent.x, y=-extent.y, z=-extent.z),
            carla.Location(x=-extent.x, y=extent.y, z=-extent.z),
            carla.Location(x=-extent.x, y=-extent.y, z=-extent.z)
        ]

        # Transform local vertices to world coordinates
        world_vertices = [vehicle_transform.transform(location) for location in local_vertices]
        vx = [vert.x for vert in world_vertices]
        vy = [vert.y for vert in world_vertices]

        return vx, vy

    def plot_path(self, ax, path):
        x_coords = [point[0] for point in path]
        y_coords = [point[1] for point in path]
        ax.plot(x_coords, y_coords, 'g', linewidth=2, label='Path Planning')
        ax.legend()



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
                collision_radius = obstacle_radius.extent.x + 1.5  # Adjust based on actual bounding box
            else:
                collision_radius = 3

            # If the obstacle is too close to the path, return False
            if obstacle_distance < collision_radius:
                # print(f"Collision detected with obstacle at ({obs_x}, {obs_y})")
                return False

        # Check if the path is within the track boundaries
        left_boundary = boundary[0]
        right_boundary = boundary[1]

        # Check both from_node and to_node
        if not self.is_within_boundaries(from_node, left_boundary, right_boundary) or \
        not self.is_within_boundaries(to_node, left_boundary, right_boundary):
            print(f"Node ({to_node.x}, {to_node.y}) is out of track boundaries.")
            return False

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
        min_x_l = min(pt.transform.location.x for pt in left_boundary)
        min_x_r = min(pt.transform.location.x for pt in right_boundary)
        max_x_l = max(pt.transform.location.x for pt in left_boundary)
        max_x_r = max(pt.transform.location.x for pt in right_boundary)
        min_x = min(min_x_l, min_x_r)
        max_x = max(max_x_l, max_x_r)
        if node_x < min_x or node_x > max_x:
            return False

        # Ensure the node is within the y-range defined by the boundaries
        min_y_l = min(pt.transform.location.y for pt in left_boundary)
        min_y_r = min(pt.transform.location.y for pt in right_boundary)
        max_y_l = max(pt.transform.location.y for pt in left_boundary)
        max_y_r = max(pt.transform.location.y for pt in right_boundary)
        min_y = min(min_y_l, min_y_r)
        max_y = max(max_y_l, max_y_r)
        if node_y < min_y or node_y > max_y:
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
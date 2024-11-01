import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import matplotlib.pyplot as plt


import random


def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):

            # # (Default) The whole map
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)


            # first quadrant
            # x = np.random.uniform(world.width/2, world.width)
            # y = np.random.uniform(world.height/2, world.height)

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

        ###############

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####
        total_weight = 0.0
        for particle in self.particles:
            read_sensor = particle.read_sensor()
            particle.weight = self.weight_gaussian_kernel(readings_robot, read_sensor)
            total_weight += particle.weight

        if total_weight == 0.0:
            print("total_weight = 0.0")
            n = len(self.particles)
            for particle in self.particles:
                particle.weight = 1.0 / n

        else:
            for particle in self.particles:
                particle.weight /= total_weight

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        
        cumulative_weights = np.cumsum([particle.weight for particle in self.particles])
        random_numbers = np.random.rand(len(self.particles))
        cnt=0
        for num in random_numbers:
            idx = np.searchsorted(cumulative_weights, num)
            cnt+=1

            particles_new.append(Particle(x = self.particles[idx].x, y = self.particles[idx].y, maze = self.world, heading = self.particles[idx].heading,sensor_limit = self.sensor_limit, weight=1, noisy = True))
        self.particles = particles_new
  
        ###############


    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
            You can either use ode function or vehicle_dynamics function provided above
        """
        ## TODO ####

        if self.control:
            for i in range(self.num_particles):
                x = self.particles[i].x
                y = self.particles[i].y
                theta = self.particles[i].heading
                for con in self.control:
                    v = con[0]
                    delta = con[1]
                    x += v * np.cos(theta) * 0.01
                    y += v * np.sin(theta) * 0.01
                    theta += delta * 0.01
                self.particles[i].x = x
                self.particles[i].y = y
                self.particles[i].heading = theta

        self.control = []


        ###############
        # pass

    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        count = 0 
        pos_errors = []
        heading_errors = []
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            while True:
                self.particleMotionModel()
                reading = self.bob.read_sensor()
                self.updateWeight(reading)
                self.resampleParticle()

                self.world.clear_objects()
                self.world.show_robot(self.bob)
                self.world.show_particles(self.particles)
                estimate_x, estimate_y, estimate_heading = self.world.show_estimated_location(self.particles)

                actual_x = self.bob.x
                actual_y = self.bob.y
                actual_heading = self.bob.heading
                pos_err = np.sqrt((estimate_x - actual_x) ** 2 + (estimate_y - actual_y) ** 2)
                diff = np.abs(estimate_heading - actual_heading * 180/np.pi) % 360
                heading_err = diff if diff <= 180 else (180 - diff)
                pos_errors.append(pos_err)
                heading_errors.append(heading_err)
                count+=1
                if count > 900:  
                    break
                # else:
                #     print(count)

            # Plot Position Error (Euclidean Distance)
            plt.figure()
            plt.plot(range(len(pos_errors)), pos_errors)
            plt.title("Position Estimation Error Over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Position Error (Euclidean Distance)")
            plt.show()

            # Plot Heading Error
            plt.figure()
            plt.plot(range(len(heading_errors)), heading_errors)
            plt.title("Orientation Estimation Error Over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Heading Error (Degrees)")
            plt.show()
                
            break  # Exit the outer loop after plotting

            ###############
import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

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

            # (Default) The whole map
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)


            ## first quadrant
            x = np.random.uniform(world.width/2, world.width)
            y = np.random.uniform(world.height/2, world.height)
            # we temporaryly picked the upper right 1/4 of the blue area of figure 3

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
            particle.weight = self.weight_gaussian_kernel(readings_robot,read_sensor)
            total_weight += particle.weight

        if total_weight == 0.0:
            n = len(self.particles)
            for particle in self.particles:
                particle.weight = 1.0 / n

        else:
            for particle in self.particles:
                particle.weight /= total_weight
        
        ###############

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####

        weights = []
        weights_sum = 0.0
        for p in self.particles:
            w = p.weight
            weights.append(w)
            weights_sum += w
        for i in range(len(weights)):
            weights[i] = weights[i] / weights_sum

        norm_weights = np.array(weights, dtype=np.float64)
        cumulative_sum = np.cumsum(norm_weights)
        for i in range(len(self.particles)):
            rnd = np.random.uniform(0, 1)
            index = 0
            for w in cumulative_sum:
                if w > rnd:
                    break
                index += 1
            particle = self.particles[index]
            particles_new.append(Particle(x = particle.x, y = particle.y, heading = particle.heading, maze = particle.maze, sensor_limit = particle.sensor_limit,  noisy = True)) # noisy = True

        ###############

        self.particles = particles_new

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
            You can either use ode function or vehicle_dynamics function provided above
        """
        ## TODO #####

        if len(self.control) == 0:
            return

        for i in range(len(self.particles):
            val = [self.particles[i].x, self.particles[i].y, self.particles[i].heading]
            for (vr, delta) in self.control:
                val[0] += vr * np.cos(val[2]) * 0.01
                val[1] += vr * np.sin(val[2]) * 0.01
                val[2] += delta * 0.01

            # update step
            self.particles[i].x = val[0]
            self.particles[i].y = val[1]
            self.particles[i].heading = val[2]
        
        self.control = []

        ###############


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        """
        The algorithm will be implemented in the runFilter function. The steps in this function are straightforward.
        You only need to constantly loop through the steps as shown in 3. Suppose p = {p1 . . . pn} are the particles
        representing the current distribution:
        def runFilter
        while True :
        sampleMotionModel (p)
        reading = vehicle_read_sensor()
        updateWeight (p , reading)
        p = resampleParticle(p)
        """
        count = 0 
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            
            self.world.clear_objects()
            self.world.show_particles(self.particles)
            self.world.show_estimated_location(self.particles)
            self.world.show_robot(self.bob)

            self.particleMotionModel()
            reading = self.bob.read_sensor()
            self.updateWeight(reading)
            self.resampleParticle()
            count += 1

            ###############

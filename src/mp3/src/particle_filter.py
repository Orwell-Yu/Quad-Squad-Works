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

        if total_weight == 0:
            n = len(self.particles)
            for particle in self.particles:
                particle.weight = 1.0 / n

        else:
            for particle in self.particles:
                particle.weight /= total_weight
        ###############
        # pass

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        
        curr_sum = 0
        cum_sum = []
        for particle in self.particles:
            curr_sum += particle.weight
            cum_sum.append(curr_sum)
        
        new_particles_indices = [np.random.uniform(0, curr_sum) for _ in range(self.num_particles)]
        new_particles_indices.sort()

        i = 0
        for j in range(self.num_particles):
            while new_particles_indices[j] > cum_sum[i]:
                i += 1
            new_particle = Particle(x=self.particles[i].x, y=self.particles[i].y, maze=self.world, sensor_limit=self.sensor_limit, weight=(1.0 / self.num_particles))  # Create a new particle object
            particles_new.append(new_particle)

        ###############

        self.particles = particles_new

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
            You can either use ode function or vehicle_dynamics function provided above
        """
        ## TODO #####
        
        for particle in self.particles:
            x = particle.x
            y = particle.y
            theta = particle.heading
            for (vr, delta) in self.control:
                dx, dy, dtheta = vehicle_dynamics(0, [x, y, theta], vr, delta)
                x += dx
                y += dy
                theta += dtheta

            particle.x = x
            particle.y = y
            particle.heading = theta

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
            
            self.particleMotionModel()
            reading = self.bob.read_sensor()
            self.updateWeight(reading)
            self.resampleParticle()
            count += 1

            ###############

import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology # type: ignore



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=100, thresh_max=255):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #2. Gaussian blur the image
        blurred_img = cv2.GaussianBlur(gray_img, (5,5), 0)
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        sobel_x = cv2.Sobel(blurred_img, cv2.CV_8U, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_img, cv2.CV_8U, 0, 1, ksize=3)
        #4. Use cv2.addWeighted() to combine the results
        sobel_combined = cv2.addWeighted(np.absolute(sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0)
        #5. Convert each pixel to unint8, then apply threshold to get binary image
        sobel_uint8 = cv2.convertScaleAbs(sobel_combined)
        binary_output = np.zeros_like(sobel_uint8, dtype=np.uint8)
        binary_output[np.logical_and(sobel_uint8 >= 20, sobel_uint8 <= thresh_max)] = 255
        # TODO: change threshold for binary output if needed

        # cv2.imshow('image', img)
        # # cv2.imshow('sobel', sobel_x.astype(np.uint8))
        # # cv2.imshow('binary', binary_output)
        # cv2.waitKey(0)

        ####

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        #2. Apply threshold on S channel to get binary image
        h,l,s = cv2.split(hls_img)
        h = cv2.convertScaleAbs(h)
        s = cv2.convertScaleAbs(s)
        l = cv2.convertScaleAbs(l)
        binary_output = np.zeros_like(s, dtype=np.uint8)
        # binary_output[np.logical_and(s >= 233, s <= 255)] = 255 # for yellow
        binary_output[np.logical_and.reduce((s <= 30, l >= 206))] = 255
        # binary_output[np.logical_and.reduce((s <= 64, l >= 128))] = 255
        #Hint: threshold on H to remove green grass
        ## TODO
        # # for main.py
        # binary_output[np.logical_and(h > 90, h < 140)] = 0 
        # binary_output[np.logical_and(h > 180, h < 300)] = 0
        binary_output[np.logical_and(h > 70, h < 140)] = 0 
        binary_output[np.logical_and(h > 180, h < 300)] = 0
        # binary_output[np.logical_and(h > 180, h < 300)] = 0
        ####

        # center = img.shape
        # center = (center[0] // 2, center[1] // 2)
        # w = 640
        # x = center[1] - w//2
        # img = img[:, x:x+w]
        # cv2.imshow('image', img)
        # cv2.waitKey(0)

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)

        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO

        ####
        binaryImage = np.zeros_like(SobelOutput, dtype=np.uint8)
        # binaryImage[(ColorOutput==255) | (SobelOutput==255)] = 255
        binaryImage[(ColorOutput==255)] = 255
        
        # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=5,connectivity=2)

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        center = img.shape
        center = (center[0] // 2, center[1] // 2)
        w = 640
        add = center[1] - w//2
        # x = center[1] - w//2
        # img = img[:, x:x+w]

        #1. Visually determine 4 source points and 4 destination points
        src_height, src_width = img.shape[:2]
        # src = np.float32([
        #                     [280, 257],
        #                     [360, 257],
        #                     [0, 400],
        #                     [597, 400],
        #                 ])
        height_p = img.shape[0] // 480
        # src = np.float32([
        #                     [280, 257 * height_p],
        #                     [375, 257 * height_p],
        #                     [0, 385 * height_p],
        #                     [597, 385 * height_p],
        #                 ])


        src = np.float32([
                            [add + 200, 216],
                            [add + 420, 216],
                            [add + 109, 314],
                            [add + 460, 314],
                        ])
        # src = np.float32([
        #                     [200, 216 * height_p],
        #                     [420, 216 * height_p],
        #                     [109, 314 * height_p],
        #                     [460, 314 * height_p],
        #                 ])
        
        des_width, des_height = src_height, src_width
        # des_width, des_height = src_width, src_height
        des = np.float32([
                        [0, 0],
                        [des_width, 0],
                        [0, des_height],
                        [des_width, des_height],
                        ])

        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        M = cv2.getPerspectiveTransform(src,des)
        Minv = cv2.getPerspectiveTransform(des,src)
        #3. Generate warped image in bird view using cv2.warpPerspective()
        warped_img = cv2.warpPerspective(img.astype(np.float32), M, (des_width, des_height))

        ## TODO
        # cv2.imshow('warped', warped_img)
        # cv2.imshow('image', img.astype(np.uint8)*255)
        # cv2.waitKey(1)

        ####

        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

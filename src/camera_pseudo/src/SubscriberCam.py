#!/usr/bin/env python
# Software License Agreement (BSD License)
# Aboniert Bilder der Wabcam
# Copyright (c) 2018 Jonas Heinke
# nach Quelle: http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
__author__ =  'Jonas Heinke <heinke.jon@gmail.com>'
__version__=  '0.1'
__license__=  'BSD'


# Python libs
import sys, time
# numpy and scipy
import numpy as np
from scipy.ndimage import filters
# OpenCV (for saving an image=
import cv2
# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError
# ROS Image message -> OpenCV2 image converter
# from cv_bridge import CvBridge, CvBridgeError
##---------------------------------------------------------------------------
from std_msgs.msg import String, Bool, Int32
# j.h keras musste nachinstalliert werden
from keras.datasets import mnist
# Instantiate CvBridge
## bridge = CvBridge()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from threading import Thread

BILD_SPEICHER = False # later
VERBOSE= False
 
# source ros.bash
# rosrun camera_pseudo SubscribeCam.py
class SubscriberCam:
	def __init__(self):
		# self.cv_bridge = CvBridge()
		print("Subscriber Cam")
  
        # subscribed Topic
		self. subscribCam = rospy.Subscriber('/camera/output/webcam/compressed_img_msgs',
										 		CompressedImage,
												self.callbackCam,
												queue_size = 1)
		
		print "subscribed from: /camera/output/webcam/compressed_img_msgs"

	def callbackCam(self, Cam):
		if VERBOSE:
			print 'CALLBECK: received image of type: "%s"' % Cam.format
		np_arr = np.fromstring(Cam.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR ) #cv2.CV_LOAD_IMAGE_COLOR
		# Zeige Bildfolge -> cv_img
		cv2.imshow('cv_img', image_np)
		cv2.waitKey(1)
   		
def main():
	# register node
	rospy.init_node('SubscriberCam', anonymous=True)
	cam=SubscriberCam()
	try:
		rospy.spin()

	except KeyboardInterrupt:
		print "Shutting down ROS SubsciberCam"
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main() #SubscribePicture()

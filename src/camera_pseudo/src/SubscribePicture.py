#!/usr/bin/env python
# Software License Agreement (BSD License)
# Aboniert ein Bild
# Copyright (c) 2018 Jonas Heinke


import numpy as np
# rospy for the subscriber
import rospy
from std_msgs.msg import String, Bool, Int32
# ROS Image message
from sensor_msgs.msg import Image, CompressedImage
## Importtyp (j.h)
## from sensor_msgs.msg import CompressedImage
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
# j.h keras musste nachinstalliert werden
from keras.datasets import mnist
# Instantiate CvBridge
bridge = CvBridge()
# image
from PIL import Image
import time
import os

BILD_SPEICHER = False

class SubscribePicture:
	def __init__(self):
		self.cv_bridge = CvBridge()
		print("Subscribe data...")


	def imageCallback(self, data):
	    rospy.loginfo(rospy.get_caller_id() + 'SubscribePicture heard  %s', data.data)
	    print("SubscribePicture received images!")
	    print(data.data)
	    print("Ende!")
	    
		# speichern als Bild zum Testen
    	if  BILD_SPEICHER:
			# os.chdir ("bilder")
			filename="daten"+zeit+".jpg"
			imageDaten=open(filename, 'w')
			imageDaten.write(data.data)
			imageDaten.close()

	def subscribe_data(self, verbose=0):
    		## Class Subsciber: http://docs.ros.org/electric/api/rospy/html/rospy.topics.Subscriber-class.html
    		rospy.Subscriber('/camera/output/specific/compressed_img_msgs',
			 	CompressedImage,
				self.imageCallback,
				queue_size = 10,
				buff_size=62720 ) ## # 6272 per picture

    		rospy.spin()  ##simply keeps python from exiting until this node is stopped

def main():
	verbose = 0  # use 1 for debug
	try:
		# register node
		rospy.init_node('SubscribePicture', anonymous=True)
		lima=SubscribePicture()
		lima.subscribe_data(verbose)
	except rospy.ROSInterruptException:
		print "Shutting down module"
		pass

if __name__ == '__main__':
	main() #SubscribePicture()

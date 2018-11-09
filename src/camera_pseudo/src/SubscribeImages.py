#!/usr/bin/env python
# Software License Agreement (BSD License)
##  j.h  basierend auf Subscriber for Images (2018-11-27)
# Copyright (c) 2018, jonas heinke j.h

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

class SubscribeImages:
	def __init__(self):
		self.cv_bridge = CvBridge()
		print("Subscribe data (Konstruktor)")
		'''
		self.subscriberImage=rospy.Subscriber(name='/camera/output/random/compressed_img_msgs',
			data_class=CompressedImage,
			callback=self.imageCallback,
			# callback_args=self.true,
			queue_size = 10,
			buff_size=62720) ## # 6272 per picture
			'''

	def saveImageFile(self,picture, nummer):
		# speichern als Bild
		# os.chdir ("bilder")
		zeit=time.strftime("%H:%M:%S")
		filename="picture_"+str(nummer.data)+"_"+zeit+".jpg"
		imageDaten=open(filename, 'w')
		imageDaten.write(picture.data)
		imageDaten.close()
		###### 
		
	def imageCallbackSubsribeNumber(self, picture):
		rospy.Subscriber(name="/camera/output/random/number",
			data_class=Int32,
			callback=self.imageNumberCallback,
			callback_args=picture,
			queue_size = 1,
			buff_size=4) ## 		

	def imageNumberCallback(self, num, picture):
		rospy.loginfo(rospy.get_caller_id() + 'SubscribeImages heard %3s. %s ',str(num), picture.data)
		print("SubscribeImages received images!")
		# print(str(num.data)+". "+picture.data) # zum TEST
                print("num.data: "+str(num.data))
		## self.saveImageFile(picture, num) # zum Test
 
 	def subscribe_image(self, verbose=0):
		#self.subscriberImage.subscrib(CompressedImage)  ## WO WIRD subscrib DEFINIERT ??
    	## Class Subsciber: http://docs.ros.org/electric/api/rospy/html/rospy.topics.Subscriber-class.html
		rospy.Subscriber(name='/camera/output/random/compressed_img_msgs',
			data_class=CompressedImage,
			callback=self.imageCallbackSubsribeNumber,
			queue_size = 1,
			buff_size=6272) ## # 6272 per picture

		rospy.spin()  ##simply keeps python from exiting until this node is stopped

def main():
	verbose = 0  # use 1 for debug
	try:
		# register node
		rospy.init_node('SubscribeImages', anonymous=True)
		subim=SubscribeImages()
		subim.subscribe_image(verbose)
	except rospy.ROSInterruptException:
		print "Shutting down module"
		pass

if __name__ == '__main__':
	main() #ListenerImage()

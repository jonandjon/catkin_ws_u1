#!/usr/bin/env python

from cv_bridge import CvBridge

import cv2
import numpy as np
import rospy
from keras.datasets import mnist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32

PUBLISH_RATE = 1 # hz
USE_WEBCAM = True


class PublishWebCam:
    def __init__(self):
        self.cv_bridge = CvBridge()

        # publish webcam
        self.publisher_webcam_comprs = rospy.Publisher("/camera/output/webcam/compressed_img_msgs",
                                                       CompressedImage,
                                                       queue_size=1)

        if USE_WEBCAM:
            self.input_stream = cv2.VideoCapture(0)
            if not self.input_stream.isOpened():
                raise Exception('Camera stream did not open\n')

        rospy.loginfo("Publishing data...")
    #--------------------------------------------------------------------------------------
   


	## veroeffentlicht Daten
    def publish_data(self, verbose=0):
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            # Note:
            # reactivate for webcam image. Pay attention to required subscriber buffer size.
            # See README.md for further information
            if USE_WEBCAM:
                self.publish_webcam(verbose)
            rate.sleep()

    def publish_webcam(self, verbose=0):
        if self.input_stream.isOpened():
            success, frame = self.input_stream.read()
            msg_frame = self.cv_bridge.cv2_to_compressed_imgmsg(frame)
            self.publisher_webcam_comprs.publish(msg_frame.header, msg_frame.format, msg_frame.data)
            if verbose:
                rospy.loginfo(msg_frame.header.seq)
                rospy.loginfo(msg_frame.format)
		
def main():
    verbose = 0  # use 1 for debug

    try:
        # register node
        rospy.init_node('PublishWebCam', anonymous=False)
        # init CameraPseudo
        cam = PublishWebCam()
        # start publishing data
        cam.publish_data(verbose)

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

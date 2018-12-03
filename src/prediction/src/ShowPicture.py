# Plot ad hoc mnist instances
import cv2
import rospy
import tensorflow as tf
import keras
import numpy as np
from keras import backend as k
from keras import callbacks	
from keras.datasets import mnist
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

file_path = "ai_train/models/weights-best.hdf5"
#load (downloaded if needed) the MNIST dataset ## muss sein
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap( ' gray ' ))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap( ' gray ' ))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap( ' gray ' ))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap( ' gray ' ))
# show the plot
plt.show()

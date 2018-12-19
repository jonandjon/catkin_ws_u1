#!/usr/bin/env python
import cv2
import rospy
import tensorflow as tf
import keras
import numpy as np
from keras import backend as k
from keras import callbacks		## ++j
from keras.datasets import mnist  ## for Test
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, String
from keras.layers import Conv2D, MaxPooling2D ##++
from keras.layers import Dense, Dropout, Flatten ## ++
from keras.models import Sequential ## ++
## j.h add ##
from PIL import Image
import time
import os
## os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import CnnModels  # mein Modul mit Klassen
## from thread import start_new_thread 
## from threading import Thread

''' Vorhersageklasse: '''
class Prediction:
    ''' Konstruktor	 '''	
    def __init__(self):
        self.cv_bridge = CvBridge()
	print("Hier ist der Konstruktor")
	## Publisher sendet Label erkannten Bildes (-> CameraPseudo)
	self.publisherPredictionNumber = rospy.Publisher("/camera/input/specific/number",
                                                         Int32,			#predictionMax - Number
                                                         queue_size=1)

	## Subscriber empfaengt validierung true|false (CameraPseudo.py <-)
	self.subscribeVerifyPrediction=rospy.Subscriber(name='/camera/output/specific/check',
								data_class=Bool,
								callback=self.callbackVerifyPrediction,
								#callback_args="subscribeVerifyPrediction",
								queue_size = 1,
								buff_size=1)


    ''' Methoden der Klasse
    ## a) Hilfsmethoden
    # Speichert ein Bild der Trainings-Menge mit Index	'''
    def saveBild(self, imageIndex):  ## for test
		(self.imagesTrain, self.labelsTrain), (self.imagesTest, self.labelsTest) = mnist.load_data()
		npArrayBild=np.expand_dims(self.imagesTest[imageIndex], axis=0) 
		label=self.labelsTest[imageIndex]
		## wandle in ein Bild um
		bild=Image.fromarray(npArrayBild[0])
		# speichert das Bild mit dem Dateinamen
		zeit=time.strftime("%H:%M:%S")
		filenameTrain="B"+str(imageIndex) +"L"+str(label)+"_"+zeit+".jpg"
		bild.save(filenameTrain)
		bild.close()
    	
    ''' * Kommunikationsmethoden
        * vom Subscriber mit dem topic: /camera/output/specific/check
          um das Validierungsergebnis zu empfangen '''
    def callbackVerifyPrediction(self, verify):
 		print("callback verify : %s" % (verify,))

''' -------------------------------------------------------------------------------------------- '''
def main():
    try:
		print("try in main")
		# register node
		rospy.init_node('prediction', anonymous=False)
		# init Prediction
		pred = Prediction()
		## Instanz der Modul.Klasse mit cnn ...
		''' Ziffern, Internetvariante '''
		### cnn=CnnModels.MnistScnn()
		''' Zffern -  HTW-Version '''
		### cnn=CnnModels.MnistCnn()
		''' Farbbilder mit 32 x 32 pixel  '''
		cnn=CnnModels.Cifar10scnn()
		##  Bild aus der Trainingsmenge wird ausgewaehlt 
		imageIndex=6 # wie in Camera Pseudo (es ist uebrigend die Ziffer 4 die zu erkennen ist)
		### Trainingsmodell, DEEP LEARNING TRAINING '''
		#?# cnn.modified()
		### Predict an Images  '''
		inputLabel, predictionLabel= cnn.predictTestImage(imageIndex)
		print("--- print in prediction.py -----------------")
		print(" Index des zu identifizierenden Bildes: %s" % (imageIndex,)) 
		print(" Label des zu identifizierenden Bildes: %s" % (inputLabel,))
		print(" Label des prediction Bild:           : %s" % (predictionLabel,))
		print("=============================================")
		# Publish your predicted number
		pred.publisherPredictionNumber.publish(predictionLabel) ## possible too direct
		
	    ## zum Test und zur Anschauung
		pred.saveBild(imageIndex)
		
	    ### Andere Variante: hole die Test und die Trainingsdaten
		#+# (imagesTrain, labelsTrain), (imagesTest, labelsTest) = cnn.loadData()
		## # Erkenne ein einzelnes Bild
		#+# predictionLabel= cnn.predictImage(imagesTest[imageIndex])

	
		rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
	main()

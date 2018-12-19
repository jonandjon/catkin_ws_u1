#!/usr/bin/env python
from __future__ import absolute_import #++
from __future__ import division		#++

from __future__ import print_function
import cv2
import rospy
import tensorflow as tf
import keras
import numpy as np
from keras import backend as k
from keras import callbacks
from keras.datasets import mnist
from keras.datasets import cifar10 # extra
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json ## https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from keras.utils import np_utils
from keras import optimizers
from keras.constraints import maxnorm ## extend
from keras.optimizers import SGD      ## extend
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
k.set_image_dim_ordering( 'th' )


# params for all classes
batch_size = 128
num_classes = 10
epochs = 12 ## 10 # 12   ## fuer Test Wert reduziert
verbose_train = 1 # 2
verbose_eval = 0
# input image dimensions
img_rows, img_cols = 28, 28

''' Simple Convolutional Neural Network for MNIST (19.4)'''
class MnistScnn:
	def __init__(self):
		print("class MnistScnn - Konstruktor")
		self.model = Sequential()
		try:
			self.loadModel()
		except:
			print("-> Modelltraining wird durchgefuehrt!")
			self.modified()
			

	'''## Load Data and convert this   '''	
	def loadData(self):
		seed = 7 		# fix random seed for reproducibility
		np.random.seed(seed)
		global X_test
		global y_test
		global X_train ##
		global y_train  ##
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		# reshape to be [samples][channels][width][height]
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
		X_test  = X_test.reshape (X_test. shape[0], 1, img_rows, img_cols).astype('float32')
		##	X_train=X_train.astype('float32')
		## 	X_test =X_test.astype('float32')
		print(X_test.shape[0]) 
		# normalize inputs from 0-255 to 0-1
		X_train = X_train / 255
		X_test  = X_test  / 255
		# one hot encode outputs
		y_train = np_utils.to_categorical(y_train)
		y_test  = np_utils.to_categorical(y_test)
		return (X_train, y_train), (X_test, y_test)

	''' 2) Define Baseline Model '''
	def scnnModel(self, num_classes):
		# create model
		## model = Sequential()
		##model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28) ))  
		self.model.add(Conv2D(32, (5, 5), input_shape=(1, img_rows, img_cols),activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.2))
		self.model.add(Flatten())
		self.model.add(Dense(batch_size, activation='relu'))
		self.model.add(Dense(num_classes, activation='softmax'))  ## softamx erst ab tensorflow==1.4 verfuegbar # softmax
		return self.model
	
	''' Speichert die Modellparameter und das Modell
	--- https://machinelearningmastery.com/save-load-keras-deep-learning-models/'''
	def saveModel(self, fileName="scnnModel"):
		### serialize model to JSON
		model_json = self.model.to_json()
		with open(fileName+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights(fileName+".h5")
		print("Saved model.h5 to disk")
		print("----------------------")
		
	''' Laedt json-Modell '''
	def loadModel(self, fileName="scnnModel"):
		json_file = open(fileName+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights(fileName+".h5")
		print("Loaded model from disk")
		return self.model
		
	
	''' Simple Convolutional Neural Network Training for MNIST (19.4) '''
	def modified(self): 
		## 1a) Load Data ##
		(X_train, y_train), (X_test, y_test)=self.loadData()
		num_classes = y_test.shape[1]
		
		## 2) Define Baseline Model # build the model
		self.model = self.scnnModel(num_classes)
		# 3) Compile model
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		## 4) Fit Model 
		self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose_train) 
		## 5) Evaluate Modell ##
		scores = self.model.evaluate(X_test, y_test, verbose=verbose_eval)
		## Speichert Modell 
		self.saveModel()
		## ----------------------------------------
		print(" Error(Sccn)  : %.2f%%" % (100-scores[1]*100))
		print(" Test accuracy: %.2f%%" % (scores[1]*100,))
		print()
		
	''' Bewertet ein einzelnes Bild aus der Testmenge '''
	def predictTestImage(self, index=6):
		## 1a) Load Dat, da ggf. noch nicht geladen ##
		(X_train, y_train), (X_test, y_test)=self.loadData()
		# expand dimension for batch
		input_data = np.expand_dims(X_test[index], axis=0)  # tensorflow
		input_label = y_test[index]
		prediction = self.model.predict(input_data)
		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)
		# output
		print("--- print in predictionTestImage() ---")
		print("index of the picture: %s" % (index,))
		print("prediction label    : %s" % (prediction,))
		print("real label          : %s" % (input_label,))
		return input_label, prediction

		
		''' Bewertet ein einzelnes Bild '''
	def predictImage(self, input_data):
		# nach eindimensional
		input_data = np.expand_dims(input_data, axis=0)  # tensorflow
		prediction = self.model.predict(input_data)
		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		# output
		print("--- print in predictionImage() ---")
		print("prediction label    : %s" % (prediction,))
		return  prediction	
		
	
""" # ##########################################################
# Vorgab-Modell des Dozenten
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU. 	
"""
class MnistCnn:
	def __init__(self):
		print("class MnistCnn - Konstruktor - HTW-Version")
		self.model = Sequential()
		try:
			self.loadModel()
		except:
			print("-> Modelltraining wird durchgefuehrt!")
			self.modified()
	

	
	'''## Load Data and convert this   '''	
	def loadData(self):
	    ## 1) Load Data ##
		
		# input image dimensions
		## img_rows, img_cols = 28, 28 siehe modul->
		# the data, split between train and test sets
		global x_test
		global y_test
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		if k.image_data_format() == 'channels_first':
		    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		    input_shape = (1, img_rows, img_cols)
		else:
		    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		    input_shape = (img_rows, img_cols, 1)

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255.
		x_test /= 255.
		print('x_train shape:', x_train.shape)
		print(x_train.shape[0], 'train samples')
		print(x_test.shape[0], 'test samples')

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes) 
		return (x_train, y_train), (x_test, y_test), input_shape
	
	''' 2) Define Baseline Model 
	# https://keras.io/layers/core/ # https://keras.io/activations/ '''
	def cnnModel(self, num_classes, input_shape):
		#-# model = Sequential()
		self.model.add(Conv2D(filters=32,
				 kernel_size=(3, 3),
				 activation='relu',
				 input_shape=input_shape))
		self.model.add(Conv2D(filters=64,
				 kernel_size=(3, 3),
				 activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(rate=0.25))
		self.model.add(Flatten())
		self.model.add(Dense(units=128, activation='relu'))
		self.model.add(Dropout(rate=0.5))
		self.model.add(Dense(units=num_classes, activation='softmax' )) # relu -> Tensor
		return self.model

		
		
	''' Speichert die Modellparameter und das Modell
	--- https://machinelearningmastery.com/save-load-keras-deep-learning-models/'''
	def saveModel(self, fileName="cnnModel"):
		### serialize model to JSON
		model_json = self.model.to_json()
		with open(fileName+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights(fileName+".h5")
		print("Saved model.h5 to disk")
		print("----------------------")
		
	''' Laedt json-Modell '''
	def loadModel(self, fileName="cnnModel"):
		json_file = open(fileName+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights(fileName+".h5")
		print("Loaded model from disk")
		return self.model

	
	''' Convolutional Neural Network Training (HTW)  '''
	def modified(self):
		## 1) Load Data ##
		(x_train, y_train), (x_test, y_test), input_shape=self.loadData()
		num_classes = y_test.shape[1]

		## 2) Define Model ##
		self.cnnModel(num_classes, input_shape)
		
		
		## 3) Compile Model ##
		self.model.compile(loss=keras.losses.categorical_crossentropy,
			      optimizer=keras.optimizers.Adadelta(),
			      metrics=['accuracy'])
		
		# callbacks
		global callbacks ##j.h## damit callbacks in Klasse keinen Fehler erzeugt 
		cb_tensorboard = callbacks.TensorBoard(write_images=True)
		# dirs and paths
		file_path = "models/weights-best.hdf5"
		cb_checkpoint = callbacks.ModelCheckpoint(
		    filepath=file_path,
		    monitor='val_acc',
		    save_best_only=True,
		    save_weights_only=False,
		    mode='max')
		callbacks = [cb_tensorboard, cb_checkpoint]
		
		## 4) Fit Model ###
		# training-
		self.model.fit(x_train,
				y_train,
				validation_data=(x_test, y_test),
				batch_size=batch_size,
				epochs=epochs,
				# callbacks=callbacks,
				verbose=verbose_train)

		## 5) Evaluate Modell ##
		score = self.model.evaluate(x_test, y_test, verbose=verbose_eval)
		## Speichert Modell 
		self.saveModel()
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

	
	''' 6) Predict Picture-Content '''	
	def predictTestImage(self, index=6): 
		## 1a) Load Data, da ggf. noch nicht geladen ##
		(_, _), (x_test, y_test),_=self.loadData()	
		# expand dimension for batch
		input_data = np.expand_dims(x_test[index], axis=0)  # tensorflow
		input_label = y_test[index]

		# example prediction call
		prediction = self.model.predict(input_data)

		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)

		# output mnist_cnn_modified
		print("Verion: mnist_cnn_modified")
		print("prediction: %s" % (prediction,))
		print("real label: %s" % (input_label,))
		return input_label, prediction
		

''' Simple Convolutional Neural Network cifar10
     Farbbilder mit 32 x 32 pixel'''   		
class Cifar10scnn:
	''' Konstruktor prueft ob Modell bereits treniert '''
	def __init__(self):
		print("- EXTEND: ")
		print("class Cifar10Scnn")
		self.model = Sequential()
		try:
			self.loadModel()
		except:
			print("-> Modelltraining wird durchgefuehrt!")
			self.modified()

	'''## Load Data and normalize this   '''	
	def loadData(self):
		seed = 7 		# fix random seed for reproducibility
		np.random.seed(seed)
		global X_test
		global y_test
		global X_train ##
		global y_train  ##
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
		# reshape to be [samples][channels][width][height]
		#-# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
		#-# X_test  = X_test.reshape (X_test. shape[0], 1, img_rows, img_cols).astype('float32')
		X_train=X_train.astype('float32')
		X_test =X_test.astype('float32')
		print(X_test.shape[0]) 
		# normalize inputs from 0-255 to 0-1
		X_train = X_train / 255.0
		X_test  = X_test  / 255.0
		# one hot encode outputs
		y_train = np_utils.to_categorical(y_train)
		y_test  = np_utils.to_categorical(y_test)
		return (X_train, y_train), (X_test, y_test)

	''' 2) Define Model '''
	def scnnModel(self, num_classes):
		## model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28) ))  
		# input image dimensions
		img_rows, img_cols = 32, 32
		batch_size =512
		self.model.add(Conv2D(32, (3, 3), input_shape=(3, img_rows, img_cols),activation='relu', kernel_constraint=maxnorm(3)))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))) ##
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Flatten())
		self.model.add(Dense(batch_size, activation='relu', kernel_constraint=maxnorm(3)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(num_classes, activation='softmax'))
		return self.model
	
	''' Speichert die Modellparameter und das Modell
	--- https://machinelearningmastery.com/save-load-keras-deep-learning-models/'''
	def saveModel(self, fileName="scnnModelCifar10"):
		### serialize model to JSON
		model_json = self.model.to_json()
		with open(fileName+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights(fileName+".h5")
		print("Saved model.h5 to disk")
		print("----------------------")
		
	''' Laedt json-Modell '''
	def loadModel(self, fileName="scnnModelCifar10"):
		json_file = open(fileName+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights(fileName+".h5")
		print("Loaded model from disk")
		return self.model
		
	
	''' Simple Convolutional Neural Network Training for MNIST (19.4) '''
	def modified(self): 
		## 1a) Load Data ##
		(X_train, y_train), (X_test, y_test)=self.loadData()
		num_classes = y_test.shape[1]
		## 2) Define Baseline Model # build the model
		self.model = self.scnnModel(num_classes)
		
		# 3) Compile model
		
		lrate = 0.01
		epochs = 25
		decay = lrate/epochs
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		self.model.summary()
		## 4) Fit Model
		batch_size=512		
		self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size) 
		## 5) Evaluate Modell ##
		verbose_eval=0
		scores = self.model.evaluate(X_test, y_test, verbose=verbose_eval)
		## Speichert Modell 
		self.saveModel()
		## ----------------------------------------
		print(" Error(scnnModelCifar10)  : %.2f%%" % (100-scores[1]*100))
		print(" Test accuracy            : %.2f%%" % (scores[1]*100,))
		print()
		
	''' Bewertet ein einzelnes Bild aus der Testmenge '''
	def predictTestImage(self, index=6):
		## 1a) Load Dat, da ggf. noch nicht geladen ##
		(X_train, y_train), (X_test, y_test)=self.loadData()
		# expand dimension for batch
		input_data = np.expand_dims(X_test[index], axis=0)  # tensorflow
		input_label = y_test[index]
		prediction = self.model.predict(input_data)
		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)
		# output
		print("--- print in predictionTestImage() ---")
		print("index of the picture: %s" % (index,))
		print("prediction label    : %s" % (prediction,))
		print("real label          : %s" % (input_label,))
		return input_label, prediction

		
		''' Bewertet ein einzelnes Bild '''
	def predictImage(self, input_data):
		# nach eindimensional
		input_data = np.expand_dims(input_data, axis=0)  # tensorflow
		prediction = self.model.predict(input_data)
		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		# output
		print("--- print in predictionImage() ---")
		print("prediction label    : %s" % (prediction,))
		return  prediction	
			


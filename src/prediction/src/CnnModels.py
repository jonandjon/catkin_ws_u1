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
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import optimizers
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
k.set_image_dim_ordering( 'th' )


# params
batch_size = 128
num_classes = 10
epochs = 1 ## 10 # 12   ## fuer Test Wert reduziert
verbose_train = 1 # 2
verbose_eval = 0

class CnnModels:
	def __init__(self):
		print("class CnnModels - Konstruktor")
		## Laden der daten
		(self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

	
	# ---------------------------------------------------- 
	# Simple Convolutional Neural Network for MNIST (19.4)
	def scnnMnist(self, index=6): 
		# fix random seed for reproducibility
		seed = 7
		np.random.seed(seed)
		# load data im Konstuktor
		## (X_train, y_train), (X_test, y_test) = mnist.load_data()
		# reshape to be [samples][channels][width][height]
		X_train = self.X_train.reshape(self.X_train.shape[0], 1, 28, 28).astype('float32')
		X_test  = self.X_test.reshape (self.X_test. shape[0], 1, 28, 28).astype('float32')
		print(X_test.shape[0]) 
		# normalize inputs from 0-255 to 0-1
		X_train = X_train / 255
		X_test  = X_test  / 255
		# one hot encode outputs
		y_train = np_utils.to_categorical(self.y_train)
		y_test  = np_utils.to_categorical(self.y_test)
		num_classes = y_test.shape[1]
		
		# define baseline model
		def scnnModel():
			# create model
			model = Sequential()
			##model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28) ))
			model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28),activation='relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.2))
			model.add(Flatten())
			model.add(Dense(128, activation='relu'))
			model.add(Dense(num_classes, activation='softmax'))  ## softamx erst ab tensorflow==1.4 verfuegbar # softmax
			# Compile model
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			## model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
			return model
		
		# build the model
		model = scnnModel()
		# Fit the model # verbose=2 || verbose_train
		model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose_train) 
		#||# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,  batch_size=200,        verbose=2)		
		scores = model.evaluate(X_test, y_test, verbose=verbose_eval) # verbose=0 || verbose_eval
		print(" Error(Sccn)  : %.2f%%" % (100-scores[1]*100))
		print(" Test accuracy: %.2f%%" % (scores[1]*100,))
		
		''' -----------------------------------------------------------------------------
		###  x) PREDICTION  ##  ## GGF. diesen Teil in eine separate Methode ##  ##  ##  '''
	
		# data to choose
		index = 6  ## temp

		# expand dimension for batch
		input_data = np.expand_dims(X_test[index], axis=0)  # tensorflow
		input_label = y_test[index]

		# example prediction call
		prediction = model.predict(input_data)

		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)

		# output
		print("prediction label: %s" % (prediction,))
		print("real label      : %s" % (input_label,))

		return input_label, prediction
	## . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
	## Bewertet ein einzelnes Bild aus der Testmenge IN WORK
	def predictTestImage(self, index=6):
		# data to choose
		## index = 6  ## temp

		# expand dimension for batch
		input_data = np.expand_dims(self.X_test[index], axis=0)  # tensorflow
		input_label = self.y_test[index]

		# example prediction call
		model = scnnModel()
		prediction = model.predict(input_data)

		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)

		# output
		print("prediction label: %s" % (prediction,))
		print("real label      : %s" % (input_label,))

		return input_label, prediction




	# ------------------------------------------------------------------------
	# Baseline Model with Multilayer Perceptrons
	def baselineMlp(self,  index=6): 
		# fix random seed for reproducibility
		seed = 7
		np.random.seed(seed)
		# load data
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		# flatten 28*28 images to a 784 vector for each image
		num_pixels = X_train.shape[1] * X_train.shape[2]
		X_train = X_train.reshape(X_train.shape[0], num_pixels)##.astype( ' float32 ' )
		X_test = X_test.reshape(X_test.shape[0], num_pixels)##.astype( ' float32 ' )
		# normalize inputs from 0-255 to 0-1
		X_train = X_train / 255
		X_test = X_test / 255
		# one hot encode outputs
		y_train = np_utils.to_categorical(y_train)
		y_test = np_utils.to_categorical(y_test)
		num_classes = y_test.shape[1]
		
		# define baseline model
		def baseline_model():
			# create model
			model = Sequential()
			
			model.add(Dense(num_pixels, input_dim=num_pixels))#, kernel_initializer= ' normal '))# ,activation= ' None ' )) ## activation= ' relu '
			model.add(Dense(num_classes))#, kernel_initializer= ' normal ' ))#, activation= ' None ' )) #activation= ' softmax ' ))
			# Compile model  optimizer=keras.optimizers.Adadelta()
			##- model.compile(loss= ' categorical_crossentropy ' ,   optimizer= ' adam ' , metrics=[ ' accuracy ' ])  ## geht nicht so
			model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
			return model
		
		# build the model
		model = baseline_model()
		# Fit the model
		model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose_train) # verbose=2 || verbose_train
		# Final evaluation of the model
		scores = model.evaluate(X_test, y_test, verbose=verbose_eval) # verbose=0 || verbose_eval
		print("Baseline Error: %.2f%%" % (100-scores[1]*100))
	
	# ------------------------------------------------------------------------------
	# Vorgab-Modell des Dozenten
	"""
	Trains a simple convnet on the MNIST dataset.

	Gets to 99.25% test accuracy after 12 epochs
	(there is still a lot of margin for parameter tuning).
	16 seconds per epoch on a GRID K520 GPU.
	"""
	def mnist_cnn_modified(self,  index=6):
		## 1) Load Data ##
		# dirs and paths
		file_path = "models/weights-best.hdf5"
		# input image dimensions
		img_rows, img_cols = 28, 28
		# the data, split between train and test sets
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

		## 2) Define Model ##
		model = Sequential()
		model.add(Conv2D(filters=32,
				 kernel_size=(3, 3),
				 activation='relu',
				 input_shape=input_shape))
		model.add(Conv2D(filters=64,
				 kernel_size=(3, 3),
				 activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(rate=0.25))
		model.add(Flatten())
		model.add(Dense(units=128, activation='relu'))
		model.add(Dropout(rate=0.5))
		## model.add(Dense(units=num_classes,activation='softmax' )) # -> Fehler softmax ungueltig
		model.add(Dense(units=num_classes, activation='softmax' )) # relu -> Tensor
		# https://keras.io/layers/core/ # https://keras.io/activations/

		## 3) Compile Model ##
		model.compile(loss=keras.losses.categorical_crossentropy,
			      optimizer=keras.optimizers.Adadelta(),
			      metrics=['accuracy'])
		

		# callbacks
		global callbacks ##j.h## damit callbacks in Klasse keinen Fehler erzeugt 
		cb_tensorboard = callbacks.TensorBoard(write_images=True)

		cb_checkpoint = callbacks.ModelCheckpoint(
		    filepath=file_path,
		    monitor='val_acc',
		    save_best_only=True,
		    save_weights_only=False,
		    mode='max')

		callbacks = [cb_tensorboard, cb_checkpoint]

		## 4) Fit Model ###
		# training-
		model.fit(x_train,
			  y_train,
			  validation_data=(x_test, y_test),
			  batch_size=batch_size,
			  epochs=epochs,
			  # callbacks=callbacks,
			  verbose=verbose_train)

		## 5) Evaluate Modell ##
		# evaluation
		score = model.evaluate(x_test, y_test, verbose=verbose_eval)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		## 6) Predict Picture-Content ##  ##  ##  ##  ##

		# data to choose
		index = 6

		# expand dimension for batch
		input_data = np.expand_dims(x_test[index], axis=0)  # tensorflow
		input_label = y_test[index]

		# example prediction call
		prediction = model.predict(input_data)

		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)

		# output mnist_cnn_modified
		print("Verion: mnist_cnn_modified")
		print("prediction: %s" % (prediction,))
		print("real label: %s" % (input_label,))
		return input_label, prediction


##################
## MODEL MODULE ##
##################
''' the purpose of this model is to create a single place
for multiple models to be defined and called. Leaving as
functions for now, I don't see a need for objects '''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils

def baseline_nn_model(strategy, number_pixels, l_rate=5e-5,
					dropout_rate=.2, layersize=None):

	default = True
	if layersize is not None:
		if len(layersize) == 3:
			default=False
			size_dense1 = layersize[0]
			size_dense2 = layersize[1]
			size_dense3 = layersize[2]
		else:
			print('incorrect sizes passed to baseline_nn_model')
			print('using defaults..');print()
	
	if default:
		size_dense1 = number_pixels
		size_dense2 = number_pixels
		size_dense3 = int(number_pixels/2)

	# create model
	with strategy.scope():
		model = Sequential()
		
		initializer_1 = tf.keras.initializers.GlorotNormal(seed=468962)
		model.add(Dense(size_dense1, input_dim=number_pixels,
						kernel_initializer=initializer_1, activation='relu'))
		model.add(Dropout(rate=dropout_rate, seed=135746))
		
		initializer_2 = tf.keras.initializers.GlorotNormal(seed=64831)
		model.add(Dense(size_dense2, kernel_initializer=initializer_2,
						activation='relu'))
		model.add(Dropout(rate=dropout_rate, seed=931576))
		
		initializer_3 = tf.keras.initializers.GlorotNormal(seed=974521)
		model.add(Dense(size_dense3,
						kernel_initializer=initializer_3, activation='relu'))
		model.add(Dropout(dropout_rate, seed=596348))
		
		initializer_4 = tf.keras.initializers.GlorotNormal(seed=579941)
		model.add(Dense(2, kernel_initializer=initializer_4,
						activation='softmax'))
		
		# compile model
		optimizer = Adam(learning_rate=l_rate)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer,
					  metrics=['acc'])
		
	return model



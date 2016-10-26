# implement a neural network for classification
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np



def fullyConnectedNet(X,Y, epochs):
	neurons = 100
	model = Sequential([
		Dense(neurons, input_dim=784, init='uniform'),
		Activation('relu'),
		Dense(neurons, init='uniform'),
		Activation('relu'),
		Dense(10, init='uniform'),
		Activation('softmax'),
	])

	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# mse does not work
	# Fit the model
	model.fit(X, y, nb_epoch=epochs, batch_size=100)

	# evaluate the model
	scores = model.evaluate(X, y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	return model

def alexNet(X, y, eochs):
	






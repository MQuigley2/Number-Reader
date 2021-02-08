#!/bin/env python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
#importing necessary libraries


#Initializing model
Model=Sequential()


#input shape is [28,28,1] for 28x28 pixel image with 1 input channel for
#brightness of black and white images.
#Convolutional 2D layer applies learned filter to every 5x5 pixel section of image
Model.add(Conv2D(filters=40, kernel_size=(5,5),padding='same', input_shape=[28,28,1],activation='relu'))
Model.add(Conv2D(filters=40, kernel_size=(5,5),padding='same',activation='relu'))


#Max pool layer reduces dimensionality by reducing 2x2 section to a single maximum value
Model.add(MaxPool2D(pool_size=(2,2)))


#Dropout layers randomly portion of inputs to zero to prevent overfitting
Model.add(Dropout(0.3))


#Second set of convolutional layers
Model.add(Conv2D(filters=80, kernel_size=(5,5),padding='same',activation='relu'))
Model.add(Conv2D(filters=80, kernel_size=(5,5),padding='same',activation='relu'))


Model.add(MaxPool2D(pool_size=(2,2)))


Model.add(Dropout(0.3))


#Flatten layer converts multidimensional tensor into vector
Model.add(Flatten())


#Network ends with fully connected layers
Model.add(Dense(units=100,activation='relu'))
Model.add(Dropout(0.3))


#Output layer has 10 units corresponding to digits 0-9
#Softmax activation ensures output values sum to 1 and can be interpereted as prabalities
Model.add(Dense(units=10,activation='softmax'))


#setting optimizer and loss function
Model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')


#Saving model 
Model.save("../models/untrainedDigitModel")

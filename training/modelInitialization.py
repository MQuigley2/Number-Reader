#!/bin/env python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
#importing necessary libraries


#Initializing model
Model=Sequential()
Model.add(Conv2D(filters=40, kernel_size=(5,5),padding='same', input_shape=[28,28,1],activation='relu'))
Model.add(Conv2D(filters=40, kernel_size=(5,5),padding='same',activation='relu'))
Model.add(MaxPool2D(pool_size=(2,2)))
Model.add(Dropout(0.2))
Model.add(Conv2D(filters=80, kernel_size=(5,5),padding='same',activation='relu'))
Model.add(Conv2D(filters=80, kernel_size=(5,5),padding='same',activation='relu'))
Model.add(MaxPool2D(pool_size=(2,2)))
Model.add(Dropout(0.2))
Model.add(Flatten())
Model.add(Dense(units=256,activation='relu'))
Model.add(Dropout(0.2))
Model.add(Dense(units=50,activation='relu'))
Model.add(Dense(units=10,activation='softmax'))
#input shape is [28,28,1] for 28x28 pixel image with 1 input channel for
#brightness of black and white images.
#Convolutional 2D layer applies learned filter to every 5x5 pixel section of image
#Max pool layer reduces dimensionality by reducing 2x2 section to a single maximum value
#Dropout layers randomly portion of inputs to zero to prevent overfitting
#Flatten layer converts multidimensional tensor into vector
#Dense layers are fully connected layers


Model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics='accuracy')
#setting optimizer and loss function


Model.save("../models/untrainedDigitModel")

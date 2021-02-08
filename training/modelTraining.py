#!/bin/env python
import numpy as np
from tensorflow import keras


#Loading pre-processed training data
X=np.load('../data/Xtrain.npy')
Y=np.load('../data/Ytrain.npy')


#Loading untrained model
Model=keras.models.load_model('../models/trainedDigitModel')


#Training model. Epochs is set to 1 for testing since training can be time consuming.
#The model used for the rest of this project was trained for 21 epochs
Model.fit(X,Y,epochs=1)


#Saving model for evaluation and use in application
Model.save('../models/trainedDigitModel')

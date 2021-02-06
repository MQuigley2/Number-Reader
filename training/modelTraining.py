#!/bin/env python
import numpy as np
from tensorflow import keras

X=np.load('../data/Xtrain.npy')
Y=np.load('../data/Ytrain.npy')
#Loading pre-processed training data

Model=keras.models.load_model('../models/trainedModel')


Model.fit(X,Y,epochs=10)
#Training model. Epochs is set to 1 for testing since training can be time consuming.
#The model used for the rest of this project was trained for 20 epochs


Model.save('../models/trainedModel')

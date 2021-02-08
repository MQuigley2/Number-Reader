#!/bin/env python
from tensorflow import keras
import numpy as np


#Loading Validation set data
X=np.load('../data/Xcval.npy')
Y=np.load('../data/Ycval.npy')


#Evaluating pre-trained model against validation set
Model=keras.models.load_model('../models/trainedDigitModel')
Model.evaluate(X,Y)

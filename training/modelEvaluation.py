#!/bin/env python
from tensorflow import keras
import numpy as np

X=np.load('../data/Xcval.npy')
Y=np.load('../data/Ycval.npy')

Model=keras.models.load_model('../models/digitModel')
Model.evaluate(X,Y)

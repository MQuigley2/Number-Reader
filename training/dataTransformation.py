#!/bin/env python
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


#loading training data to pandas dataframe
df_all=pd.read_csv('../data/trainingData.csv')

#splitting data into training and validation sets
df_cval=df_all.iloc[36000:]
df_train=df_all.iloc[:36000]


#converting label column to numpy array
Ytrain=df_train['label'].to_numpy()
Ycval=df_cval['label'].to_numpy()


#converting pixel values to numpy arrays
#reshaping to match input shape for convolutional neural network
#dividing by maximum pixel brightness for normalization
Xtrain=df_train.drop(columns='label').to_numpy().reshape([-1,28,28,1])/255
Xcval=df_cval.drop(columns='label').to_numpy().reshape([-1,28,28,1])/255


#converts array of numbers 0-9 into array of 1 by 10 arrays
#a label of 2 would be converted to an array with value 1 at index 2 and zero at all other indexes
YtrainCategorical=to_categorical(Ytrain)
YcvalCategorical=to_categorical(Ycval)



#saving processed data
np.save('../data/Xtrain.npy',Xtrain)
np.save('../data/Ytrain.npy',YtrainCategorical)
np.save('../data/Xcval.npy',Xcval)
np.save('../data/Ycval.npy',YcvalCategorical)

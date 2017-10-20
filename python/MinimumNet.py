from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Flatten
from keras.optimizers import Adam

import math
import os
import numpy as np


DIM=3
BATCH_SIZE=10


def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128))
    return model

def train():
    model=generator_model(10,24)
    for epoch in range(100):
        noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
        noise2= np.array([np.random.uniform(0, 0, 100) for _ in range(BATCH_SIZE)])
        loss=model.train_on_batch(noise,noise2)
        print(loss)
    model.save_weights('test.h5')
    return model




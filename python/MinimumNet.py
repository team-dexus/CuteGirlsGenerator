from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Flatten
from keras.optimizers import Adam
DIM=3


def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(width*height*DIM))
    model.add(Activation('tanh'))
    model.add(Dense(4*width*height*DIM))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((height, width, DIM, 4), input_shape=(4*width*height*DIM,)))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(4, (2, 2, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(1, (2, 2, 1), padding='same'))
    model.add(Activation('tanh'))
    return model


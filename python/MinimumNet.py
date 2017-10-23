from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Flatten
from keras.optimizers import Adam
import numpy as np
DIM=3


def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
def train():
    g=generator_model(24,40)
    g_opt = Adam(lr=1e-5, beta_1=0.1)
    g.compile(loss='binary_crossentropy', optimizer=g_opt)
    noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(1)])
    g.train_on_batch(noise, np.array([[1]]))
    noise = np.array([np.random.uniform(0, 0, 100) for _ in range(1)])
    print(g.predict(noise))
    g.save_weights('MinNet.h5')
    with open('generator.json', 'w') as f:
        f.write(g.to_json())
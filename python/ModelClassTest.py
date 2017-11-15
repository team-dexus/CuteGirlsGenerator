from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D,UpSampling2D
from keras.layers.convolutional import Conv3D, MaxPooling3D ,Conv2D, MaxPooling2D
from keras.layers.core import Flatten,Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
DIM=3

def generator_model(width,height):
    noise = Input(shape=(100,))
    x = Dense(input_dim=100, output_dim=128,bias_initializer='he_uniform')(noise)
    x = Activation('tanh')(x)
    x = Dense(width*height)(x)
    x = Activation('tanh')(x)
    x = Dense(width*height*4)(x)
    x = Activation('tanh')(x)
    x = Dense(width*4*height*4*4)(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Reshape((height*4, width*4,4), input_shape=(width*4*height*4*4,))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Conv2D(DIM, (3, 3), padding='same')(x)
    output = Activation('tanh')(x)
    model = Model(inputs=noise, outputs=output)
    return model

'''
def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128,bias_initializer='he_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(width*height))
    model.add(Activation('tanh'))
    model.add(Dense(width*height*4))
    model.add(Activation('tanh'))
    model.add(Dense(width*4*height*4*4))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    model.add(Reshape((height*4, width*4,4), input_shape=(width*4*height*4*4,)))
    
    #model.add(Dropout(0.5))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2D(DIM, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    return model
'''

def discriminator_model(width,height):
    illust=Input(shape=(height,width,DIM,))
    x = Conv2D(16, (5, 5), padding='same' , input_shape=(height, width, DIM))(illust)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (5, 5),padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5),padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)
    model = Model(inputs=illust, outputs=output)
    
    return model
'''
def discriminator_model(width,height):
    model = Sequential()
    model.add(
            Conv2D(16, (5, 5),
            padding='same',
            input_shape=(height, width, DIM))
            )
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5),padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(LeakyReLU(0.2))
    model.add(Dense(2048))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

'''
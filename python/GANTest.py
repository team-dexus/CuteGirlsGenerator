from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import Conv2D, Conv3D, MaxPooling2D
from keras.layers.core import Flatten



import math
import os
import numpy as np
import glob
from PIL import Image
TRAIN_IMAGE_PATH="/Users/KOKI/Documents/TrainData2/*"
GENERATED_IMAGE_PATH="/Users/KOKI/Documents/Generated/"

def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(4*width*height*3))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((height, width,  3, 4), input_shape=(4*width*height*3,)))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(4, (2, 2, 2), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(1, (2, 2, 2), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
    
def data_import(width,height):
    image = np.empty((0,height,width,3), dtype=np.uint8)
    list=glob.glob(TRAIN_IMAGE_PATH)
    for i in list:
        im_reading = np.array( Image.open(i).resize((width,height)))
        #im_reading = im_reading.transpose(1,0,2)
        image = np.append(image, [im_reading], axis=0)
    return image

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((width*cols, height*rows,3),
                              dtype=generated_images.dtype)
    #coreturn combined_image

    for index, image in enumerate(generated_images):
        i = index % cols
        j = int(index/cols)
        combined_image[width*i:width*(i+1), height*j:height*(j+1),0:3] = image[:,:,0:3 ]
    return combined_image
    
def save_images(images,file_name):
    if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
    Image.fromarray(images.astype(np.uint8))\
        .save(GENERATED_IMAGE_PATH+"%s.png" % (file_name))

def save_generated_image(image,name):
    Imag=combine_images(image)
    save_images(Imag,name)

BATCH_SIZE=128
noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
g=generator_model(40,25)
image=g.predict(noise,verbose=1)
image=image*127.5+127.5
image=image.reshape(image.shape[0:4])
save_generated_image(image,"IWannaFuckCuteLittleGirls")

    
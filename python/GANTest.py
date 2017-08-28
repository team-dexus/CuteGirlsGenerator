#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:43:21 2017

@author: KOKI
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import Conv3D, Convolution2D,Conv2D,MaxPooling2D,MaxPooling3D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout


import math
import os
import numpy as np
import glob
from PIL import Image
TRAIN_IMAGE_PATH="/Users/KOKI/Documents/TrainData2/*"
GENERATED_IMAGE_PATH="/Users/KOKI/Documents/Generated/"

def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(4*width*height*3))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape(( height, width, 3,4), input_shape=(4*width*height*3,)))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(4, (2, 2, 2), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(1, (2, 2, 2), padding='same'))
    model.add(Activation('tanh'))
    return model



def discriminator_model(width,height):
    model = Sequential()
    model.add(
            Conv2D(64, (16, 16),
            padding='same',
            input_shape=(width, height,3))
            )

    model.add(Activation('tanh'))
    model.add(Conv2D(128,(5,5)))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    
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
    Imag = Imag*127.5 + 127.5
    save_images(Imag,name)

def train(width,height):
    BATCH_SIZE = 32
    NUM_EPOCH = 20
    WIDTH,HEIGHT=width,height
    X_train=data_import(WIDTH,HEIGHT)
    X_train = (X_train.astype(np.float32) - 127.5)/127.5

    d = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    d.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator （discriminator部分の重みは固定）
    d.trainable = False
    g = generator_model()
    dcgan = Sequential([g, d])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)

            # 生成画像を出力
            if index % 500 == 0:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8))\
                    .save(GENERATED_IMAGE_PATH+"%04d_%04d.png" % (epoch, index))

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = d.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

        g.save_weights('generator.h5')
        d.save_weights('discriminator.h5')

#BATCH_SIZE=256
#noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
#g=generator_model(40,25)
#d=discriminator_model(160,100)
#image=g.predict(noise)
#g_loss=d
#image=image.reshape(image.shape[0:4])
#save_generated_image(image,"test")

    
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
import glob
from PIL import Image
TRAIN_IMAGE_PATH="/Users/KOKI/Documents/TrainData3/*" 
GENERATED_IMAGE_PATH="/Users/KOKI/Documents/Generated/" 
BATCH_SIZE = 13
NUM_EPOCH = 2000
DIM=3

def generator_model(width,height):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(width*height))
    model.add(Activation('tanh'))
    model.add(Dense(width*height*DIM))
    model.add(Activation('tanh'))
    model.add(Dense(2*width*height*DIM))
    model.add(Activation('tanh'))
    model.add(Dense(4*width*height*DIM))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((height, width, DIM, 4), input_shape=(4*width*height*DIM,)))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(4, (2, 2, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(Conv3D(2, (2, 2, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv3D(1, (2, 2, 1), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model(width,height):
    model = Sequential()
    model.add(
            Conv3D(16, (5, 5, 5),
            padding='same',
            input_shape=(height, width, DIM, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Conv3D(32, (5, 5, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
    
def data_import(width,height):
    image = np.empty((0,height,width,DIM), dtype=np.uint8)
    list=glob.glob(TRAIN_IMAGE_PATH)
    for i in list:
        im_reading = Image.open(i).resize((width,height))
        im_reading = im_reading.convert("RGB")
        im_reading = np.array( im_reading)
        print(im_reading.shape)
        #im_reading = im_reading.transpose(1,0,2)
        print(i)
        image = np.append(image, [im_reading], axis=0)
        
        
    return image

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((width*cols, height*rows,DIM),
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
    
    

def train(width,height):
    WIDTH,HEIGHT=width,height
    X_train=data_import(WIDTH,HEIGHT)
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0:4])

    d = discriminator_model(WIDTH,HEIGHT)
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    d.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator （discriminator部分の重みは固定）
    d.trainable = False
    g = generator_model(round(WIDTH/4),round(HEIGHT/4))
   
    try:
        g.load_weights('generator.h5')
    except:
        print("generator couldn't load")
    try:
        d.load_weights('discriminator.h5')
    except:
        print("discriminator couldn't load")
    
   
    dcgan = Sequential([g, d])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    d_loss=0
    g_loss=0
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=1)
            generated_images=generated_images.reshape(generated_images.shape[0:4])
            # 生成画像を出力
            
            if epoch % 100 == 0:

                generated_images=generated_images*127.5+127.5
                save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))
                generated_images=(generated_images-127.5)/127.5
            
            
            # discriminatorを更新
            print(image_batch.shape)
            print(generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            X=X.reshape(X.shape+(1,))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            if 1>g_loss-g_loss:
                d_loss = d.train_on_batch(X, y)

            # generatorを更新 
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))
        

        
        g.save_weights('generator.h5')
        with open('generator.json', 'w') as f:
            f.write(g.to_json())
        d.save_weights('discriminator.h5')

    generated_images=generated_images*127.5+127.5
    save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))

#train(24,40)
#test=data_import(24,40)
#save_generated_image(test,"sex is life")

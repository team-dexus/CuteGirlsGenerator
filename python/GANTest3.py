from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense ,Input
from keras.layers import Reshape,concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D,UpSampling2D
from keras.layers.convolutional import Conv3D, MaxPooling3D ,Conv2D, MaxPooling2D
from keras.layers.core import Flatten,Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


import math
import os
import numpy as np
import glob
from PIL import Image
import pickle

import sys
sys.path.append("/Users/KOKI/PerfectMakeGirls/python/i2v")
import i2v


TRAIN_IMAGE_PATH="/Users/KOKI/Documents/TrainData3/*" 
GENERATED_IMAGE_PATH="/Users/KOKI/Documents/Generated/" 
BATCH_SIZE = 200
NUM_EPOCH = 10
DIM=3
NUMBER_OF_TAG=1539
'''
try:
    with open('illust2vec.pickle', 'r') as f:
        illust2vec = pickle.load(f)
except:
    illust2vec = i2v.make_i2v_with_chainer(
    "./i2v/illust2vec_tag_ver200.caffemodel", "./i2v/tag_list.json")
    with open('illust2vec.pickle', 'wb') as f:
        pickle.dump(illust2vec,f)
'''
    

def generator_model(width,height):
    noise = Input(shape=(100,))
    tags_input = Input(shape=(NUMBER_OF_TAG,))
    tags = Dense(input_dim=NUMBER_OF_TAG, output_dim=128,bias_initializer='glorot_normal')(tags_input)
    tags = Activation('tanh')(tags)
    tags = Dense(128,bias_initializer='glorot_normal')(tags)
    tags = Activation('tanh')(tags)
    tags = Dense(width*height,bias_initializer='glorot_normal')(tags)
    tags = Activation('tanh')(tags)
    x = Dense(input_dim=100, output_dim=128,bias_initializer='glorot_normal')(noise)
    x = Activation('tanh')(x)
    x = Dense(width*height,bias_initializer='glorot_normal')(x)
    x = Activation('tanh')(x)
    x = concatenate([x, tags])
    x = Dense(width*height*4,bias_initializer='glorot_normal')(x)
    x = Activation('tanh')(x)
    x = Dense(width*4*height*4*4,bias_initializer='glorot_normal')(x)
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
    x = Dropout(0.5)(x)
    output = Activation('tanh')(x)
    model = Model(inputs=[noise,tags_input], outputs=output)
    return model


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
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(2048,bias_initializer='glorot_normal')(x)
    x = LeakyReLU(0.2)(x)
    tag_output=Dense(1024,bias_initializer='glorot_normal')(x)
    tag_output=Dense(NUMBER_OF_TAG,bias_initializer='glorot_normal')(tag_output)
    tag_output = Activation('sigmoid')(tag_output)
    x = Dense(256,bias_initializer='glorot_normal')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1,bias_initializer='glorot_normal')(x)
    output = Activation('sigmoid')(x)
    model = Model(inputs=illust, outputs=[output,tag_output])
    
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
        
    try:
        with open('tags.pickle', 'rb') as f:
            estimated_tags = pickle.load(f)
            print("load from pickle")
        
    except:
        
        try:
            with open('illust2vec.pickle', 'rb') as f:
                illust2vec = pickle.load(f)
                print("pickle i2v")
        except:
            illust2vec = i2v.make_i2v_with_chainer(
            "./i2v/illust2vec_tag_ver200.caffemodel", "./i2v/tag_list.json")
            with open('illust2vec.pickle', 'wb') as f:
                pickle.dump(illust2vec,f)
            print("new i2v")
        
        batch_size=5
        estimated_tags=np.zeros((0,NUMBER_OF_TAG))
        for i in range(math.floor(len(image)/batch_size+1)):
            print(str(i)+"/"+str(math.floor(len(image)/batch_size+1)))
            if len(image)<batch_size*(i+1):
                batch=np.array(image[batch_size*i:])
            else:
                batch=np.array(image[batch_size*i:batch_size*(i+1)])
            print(estimated_tags.shape)
            
            estimated_tags=np.append(estimated_tags,illust2vec.extract_feature(batch),axis=0) 
            
        with open('tags.pickle', 'wb') as f:
            pickle.dump(estimated_tags,f)

    return image,estimated_tags

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
    X_train,tags=data_import(WIDTH,HEIGHT)
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
    
    
    noise=Input(shape=(100,))
    tag=Input(shape=(NUMBER_OF_TAG,))
    fake=g([noise,tag])
    fake,estimated_tag=d(fake)
    dcgan = Model(inputs=[noise,tag],outputs=[fake,estimated_tag])
    #dcganモデルを作成
    
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
            tag_batch = tags[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict([noise,tag_batch], verbose=1)
            
            #generated_images=generated_images.reshape(generated_images.shape[0:4])
            # 生成画像を出力
            
            if index==0:
                generated_images=generated_images*127.5+127.5
                save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))
                generated_images=(generated_images-127.5)/127.5
            
            
            # discriminatorを更新
            print(image_batch.shape)
            print(generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            #X=X.reshape(X.shape+(1,))
            y = [np.array([1]*BATCH_SIZE+[0]*BATCH_SIZE),np.concatenate((tag_batch, tag_batch))]
            #return y
            #if 1>g_loss-d_loss:
            d_loss = d.train_on_batch(X, y)

            # generatorを更新 
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch([noise,tag_batch], [np.array([1]*BATCH_SIZE),tag_batch])
            print("epoch: %d, batch: %d, g_loss: %s, d_loss: %s" % (epoch, index, str(g_loss), str(d_loss)))
        

        
        g.save_weights('generator.h5')
        with open('generator.json', 'w') as f:
            f.write(g.to_json())
        d.save_weights('discriminator.h5')

    generated_images=generated_images*127.5+127.5
    save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))

def generate_test_image(image_name):
    g = generator_model(round(24/4),round(40/4))
   
    try:
        g.load_weights('generator.h5')
    except:
        print("generator couldn't load")
     
    d = discriminator_model(24,40)
    try:
        d.load_weights('discriminator.h5')
    except:
        print("discriminator couldn't load")
        
    noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
    generated_images = g.predict(noise, verbose=1)
    loss=d.predict(generated_images)
    #generated_images=generated_images.reshape(generated_images.shape[0:4])
    generated_images=generated_images*127.5+127.5
    save_generated_image(generated_images,image_name)
    generated_images=(generated_images-127.5)/127.5
    return loss
    

#train(24,40)
#test=data_import(24,40)
#save_generated_image(test,"sex is life")

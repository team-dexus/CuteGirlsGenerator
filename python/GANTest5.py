from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense ,Input
from keras.layers import Reshape,multiply,add,concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D,UpSampling2D
from keras.layers.convolutional import Conv3D, MaxPooling3D ,Conv2D, AveragePooling2D,Conv2DTranspose
from keras.layers.core import Flatten,Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU ,ELU
import keras.backend as K
from keras.losses import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator


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
BATCH_SIZE = 10
NUM_EPOCH = 1000
DIM=3
NUMBER_OF_TAG=1539
'''
try:
    with open('illust2vec.pickle', 'r') as f:
        illust2vec = pickle.load(f)
except:
    illust2vec = i2v.make_i2v_with_chainer(
    "./i2v/illust2vec_tag_ver200.caffemodel", "./i2v/tag_list.json")2
    with open('illust2vec.pickle', 'wb') as f:
        pickle.dump(illust2vec,f)
'''
    

def generator_model(width,height):#144:240
    noise_input = Input(shape=(100,))
    noise = Dense(100)(noise_input)
    tags_input = Input(shape=(NUMBER_OF_TAG,))
    tags = Dense(input_dim=NUMBER_OF_TAG, output_dim=100)(tags_input)
    tags = ELU()(tags)
    x = concatenate([noise, tags])
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dense(int(width/16*height/16*128),use_bias=False)(x)
    x = Reshape((int(height/16),int(width/16),128), input_shape=(int(width/16*height/16*128),))(x)
    
   
    x = Conv2DTranspose(256, (3,3) ,strides=2,padding='same', use_bias=False,kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(256, (3,3) ,strides=2,padding='same', use_bias=False,kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(128, (3,3) ,strides=2,padding='same', use_bias=False,kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(64, (3,3) ,strides=2,padding='same', use_bias=False,kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    x = Conv2DTranspose(DIM, (5,5) ,strides=1,padding='same',kernel_initializer='he_normal')(x)
    output = Activation('tanh')(x)
    model = Model(inputs=[noise_input,tags_input], outputs=output)
    
    return model


def discriminator_model(width,height):
    illust=Input(shape=(height,width,DIM,))
    x = Conv2D(32, (7, 7), padding='same',kernel_initializer='he_normal',use_bias=False)(illust)
    x = ELU()(x)
    x = AveragePooling2D((2,2))(x)
    x = Conv2D(64, (5, 5), padding='same',kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D((2,2))(x)
    x = Conv2D(128, (3, 3), padding='same' ,kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D((2,2))(x)
    x = Conv2D(256, (3, 3), padding='same' ,kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D((2,2))(x)
    x = Conv2D(256, (3, 3), padding='same' ,kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D((2,2))(x)
    x = Flatten()(x)
    
    tag_output=Dense(NUMBER_OF_TAG,kernel_initializer='glorot_normal')(x)
    tag_output = Activation('sigmoid')(tag_output)
    x = Dense(1,kernel_initializer='glorot_normal')(x)
    output = Activation('linear')(x)
    model = Model(inputs=illust, outputs=[output,tag_output])
    
    return model

 
    
def data_import(width,height):
    try:
        with open('images.pickle', 'rb') as f:
            image = pickle.load(f)
            print("image load from pickle")
    except:
        image = np.empty((0,height,width,DIM), dtype=np.uint8)
        list=glob.glob(TRAIN_IMAGE_PATH)
        for i in list:
            im_reading = Image.open(i).resize((width,height))
            if im_reading.mode=="RGB":
                im_reading = np.array( im_reading)
                
            else: 
                im_reading = im_reading.convert("RGB")
                im_reading = np.array( im_reading)
                for j in range(len(im_reading)):
                    for k in range(len(im_reading[0])):
                        if np.all(im_reading[j][k]==[71,112,76]) or np.all(im_reading[j][k]==[0,0,0]) or np.all(im_reading[j][k]==[76,105,113]):
                            im_reading[j][k]=[255,255,255]
    
                                  
                            #RGB (71,112,76),(75,105,113)のための例外処理。なんかいい方法があったら言ってくれ
            print(i)
            
            #im_reading = im_reading.transpose(1,0,2)
            image = np.append(image, [im_reading], axis=0)
        
            with open('images.pickle', 'wb') as f:
                pickle.dump(image,f) 
        print("new image")
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
    
    

def train(width,height,test=False):
    WIDTH,HEIGHT=width,height
    X_train,tags=data_import(WIDTH,HEIGHT)
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=True)
    datagen=datagen.flow(X_train,tags,batch_size=BATCH_SIZE)
    save_generated_image(datagen.next()[0],"generated")
    print("sex!")
    '''
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0:4])
    '''

    d = discriminator_model(WIDTH,HEIGHT)
    d_opt = Adam(lr=1e-4, beta_1=0.1)
    d.compile(loss='mean_squared_error', optimizer=d_opt)

    # generator+discriminator （discriminator部分の重みは固定）
    d.trainable = False
    g = generator_model(WIDTH,HEIGHT)
   
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
    dcgan.compile(loss='mean_squared_error', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    d_loss=0
    g_loss=0
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            batch = datagen.next()
            image_batch = (batch[0].astype(np.float32) - 127.5)/127.5
            tag_batch = batch[1]
            noise = np.random.normal(0, 0.5, [len(image_batch),100])
            
            generated_images = g.predict([noise,tag_batch], verbose=1)
            
            #generated_images=generated_images.reshape(generated_images.shape[0:4])
            # 生成画像を出力
            
            if index%10==0:
                generated_images=generated_images*127.5+127.5
                save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))
                generated_images=(generated_images-127.5)/127.5
            
            
            # discriminatorを更新
            #print(image_batch.shape)
            #print(generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            #X=X.reshape(X.shape+(1,))
            y = [np.array([1]*len(image_batch)+[0]*len(image_batch)),np.concatenate((tag_batch, tag_batch))]
            #return y
            #if 1>g_loss-d_loss:
            d_loss = d.train_on_batch(X, y)

            # generatorを更新 
            print(len(image_batch))
            noise = np.random.normal(0, 0.5, [len(image_batch),100])
            test=dcgan.predict([noise,tag_batch])
            g_loss = dcgan.train_on_batch([noise,tag_batch], [np.array([1]*len(image_batch)),tag_batch])
            
            print(test)
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

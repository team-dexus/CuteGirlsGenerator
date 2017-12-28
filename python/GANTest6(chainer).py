import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import math
import os
import glob
import pickle
from PIL import Image

from network import generator,discriminator

TRAIN_IMAGE_PATH="/Users/KOKI/Documents/TrainData3/*" 
GENERATED_IMAGE_PATH="/Users/KOKI/Documents/Generated/" 
BATCH_SIZE = 10
NUM_EPOCH = 1000
DIM=3
NUMBER_OF_TAG=1539



def check_accuracy(model, xs, ts):
    ys = model(xs)
    loss = F.softmax_cross_entropy(ys, ts)
    ys = np.argmax(ys.data, axis=1)
    cors = (ys == ts)
    num_cors = sum(cors)
    accuracy = num_cors / ts.shape[0]
    return accuracy, loss

def data_import(width,height):
    try:
        with open('images.pickle', 'rb') as f:
            image = pickle.load(f)
            print("image load from pickle")
    except:
        image = np.empty((0,height,width,DIM), dtype=np.uint8)
        list=glob.glob(TRAIN_IMAGE_PATH)
        for i in list:
            im_reading = Image.open(i)
            im_reading .thumbnail((width, height),Image.ANTIALIAS)
            print(im_reading.size)
            bg = Image.new("RGBA",[width,height],(255,255,255,255))
            bg.paste(im_reading,(int((width-im_reading.size[0])/2),int((height-im_reading.size[1])/2)))
            im_reading=bg.copy()
            #im_reading.show()
            
            if im_reading.mode=="RGB":
                im_reading = np.array(im_reading)
                
            else: 
                im_reading = im_reading.convert("RGB")
                im_reading = np.array(im_reading)
                for j in range(len(im_reading)):
                    for k in range(len(im_reading[0])):
                        if np.all(im_reading[j][k]==[71,112,76]) or np.all(im_reading[j][k]==[0,0,0]) or np.all(im_reading[j][k]==[76,105,113]):
                            im_reading[j][k]=[255,255,255]
    
                                  
                            #RGB (71,112,76),(75,105,113)のための例外処理。なんかいい方法があったら言ってくれ

            print(i)
            
            #im_reading = im_reading.transpose(1,0,2)
            print(im_reading.shape)
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

def train(width,height):
    g = generator(width, height, 100)
    d = discriminator()

    g_opt = chainer.optimizers.Adam()
    g_opt.setup(g)
    d_opt = chainer.optimizers.Adam()
    d_opt.setup(d)

    X_train,tags=data_import(width,height)
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.transpose(0,3,1,2)
    print(X_train.shape)
    tags = tags.astype(np.float32)


    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    
    for epoch in range(100):

        for index in range(num_batches):
            '''
            x = xs[(j * bm):((j + 1) * bm)]
            t = ts[(j * bm):((j + 1) * bm)]
            '''
            image_batch=X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            tag_batch=tags[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
            noise=np.random.normal(0, 0.5, [len(image_batch),100])
            z = Variable(noise.astype(np.float32))
            
        
            x = g(z,tag_batch)
            if index==0:
                generated_images=x.data*127.5+127.5
                generated_images=generated_images.transpose(0,2,3,1)
                save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))
            print(x.shape)
            yl = d(x,tag_batch)
            print(yl.data)

            g_loss=F.mean_squared_error(yl, Variable(np.ones((len(image_batch),1), dtype=np.float32)))
            d_loss=F.mean_squared_error(yl, Variable(np.zeros((len(image_batch),1), dtype=np.float32)))
            
            yl2 = d(image_batch,tag_batch)
            d_loss+=F.mean_squared_error(yl2, Variable(np.ones((len(image_batch),1), dtype=np.float32)))
            print(yl2.data)

            g.cleargrads()
            g_loss.backward()
            g_opt.update()
            
            d.cleargrads()
            d_loss.backward()
            d_opt.update()
            print("epoch %d, batch: %d, g_loss: %f, d_loss: %f" %(epoch,index,g_loss.data,d_loss.data))
        
        serializers.save_npz('generator.model', g)
        serializers.save_npz('discriminator.model', d)

        #print("Epoch %d loss(train) = %f, accuracy(train) = %f, accuracy(test) = %f" % (i + 1, loss_train.data, accuracy_train, accuracy_test))
        


if __name__ == '__main__':
    train(512,512)
    print("sex")
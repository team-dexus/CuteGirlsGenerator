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

#from network import generator,discriminator
from network2 import generator,discriminator


TRAIN_IMAGE_PATH="/Users/KOKI/Documents/TrainData5/*" 
GENERATED_IMAGE_PATH="/Users/KOKI/Documents/Generated/" 
BATCH_SIZE = 10
NUM_EPOCH = 10
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
        with open('images%d.pickle'%width, 'rb') as f:
            image = pickle.load(f)
            print("image load from pickle")
    except:
        image = np.empty((0,height,width,DIM), dtype=np.uint8)
        list=sorted(glob.glob(TRAIN_IMAGE_PATH))
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
        
            with open('images%d.pickle'%width, 'wb') as f:
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

def train(width,height,depth,start_alpha=0):
    g = generator(512, 512, 100)
    try:
        serializers.load_npz("generator.model", g)
        print("generator loaded")
    except:
        pass
    d = discriminator()
    try:
        serializers.load_npz("discriminator.model", d)
        print("discriminator loaded")
    except:
        pass

    g_opt = chainer.optimizers.Adam(alpha=0.001, beta1=0.0, beta2=0.99)
    g_opt.setup(g)
    g_opt.add_hook(chainer.optimizer.WeightDecay(0.0005))
    
    d_opt = chainer.optimizers.Adam(alpha=0.001, beta1=0.0, beta2=0.99)
    d_opt.setup(d)
    d_opt.add_hook(chainer.optimizer.WeightDecay(0.0005))

    X_train,tags=data_import(16*(2**depth),16*(2**depth))
    '''
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.transpose(0,3,1,2)
    '''
    print(X_train.shape)
    tags = tags.astype(np.float32)


    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    alpha=start_alpha
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            if alpha<1.0:
                alpha=alpha + 5e-4
            '''
            x = xs[(j * bm):((j + 1) * bm)]
            t = ts[(j * bm):((j + 1) * bm)]
            '''
            image_batch=X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            image_batch = (image_batch.astype(np.float32) - 127.5)/127.5
            image_batch = image_batch.transpose(0,3,1,2)
            tag_batch=tags[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
            noise=np.random.normal(0, 0.5, [len(image_batch),100])
            z = Variable(noise.astype(np.float32))
            
        
            x = g(z,tag_batch,depth,alpha)
            if index%10==0:
                generated_images=x.data*127.5+127.5
                generated_images=generated_images.transpose(0,2,3,1)
                save_generated_image(generated_images,"%04d_%04d.png" % (epoch, index))

            yl= d(x,tag_batch,depth,alpha)
            #print(yl)
            #g_loss=F.mean_squared_error(yl, Variable(np.ones((len(image_batch),1), dtype=np.float32)))
            #d_loss=F.mean_squared_error(yl, Variable(np.zeros((len(image_batch),1), dtype=np.float32)))

            yl2= d(image_batch,tag_batch,depth,alpha)
            #print(yl2)
            #d_loss+=F.mean_squared_error(yl2, Variable(np.ones((len(image_batch),1), dtype=np.float32)))
            d_loss=-F.sum(yl2 - yl) / len(image_batch)
            d_loss+=F.mean(0.001*yl*yl)
            g_loss=-F.sum(yl) / len(image_batch)
            '''
            mean=F.mean(x,axis=0)
            dev=x-F.broadcast_to(mean, x.shape)
            devdev=dev*dev
            var=F.mean(devdev)
            
            g_loss-= var
            '''
            

            g.cleargrads()
            g_loss.backward()
            g_opt.update()
            
            d.cleargrads()
            d_loss.backward()
            d_opt.update()
            print("epoch %d, batch: %d, g_loss: %f, d_loss: %f, alpha: %f, depth: %d" %(epoch,index,g_loss.data,d_loss.data,alpha,depth))
        
        serializers.save_npz('generator.model', g)
        serializers.save_npz('discriminator.model', d)

        #print("Epoch %d loss(train) = %f, accuracy(train) = %f, accuracy(test) = %f" % (i + 1, loss_train.data, accuracy_train, accuracy_test))
        


if __name__ == '__main__':
    #start=3
    #train(512,512,start,0.3)
    start=0
    for i in range(6-start):
        train(512,512,i+start)
    
    
    print("sex")
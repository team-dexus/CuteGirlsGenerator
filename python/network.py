import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

NUMBER_OF_TAG=1539

class generator(chainer.Chain):
    def __init__(self, width, height, z_size):
        super().__init__(
            l1=L.Linear(z_size, 100),
            l2=L.Linear(NUMBER_OF_TAG, 100),
            l3=L.Linear(200, 100),
            l4 = L.Linear(100, int(width/32*height/32*128)),
            
            #bn_l3=L.BatchNormalization(100),
            #bn_l4=L.BatchNormalization(100),
           
            dc1=L.Deconvolution2D(None, 256, 4, stride=2, pad=1, nobias=True),
            dc2=L.Deconvolution2D(256, 256, 3, stride=1, pad=1, nobias=True),
            dc3=L.Deconvolution2D(256, 256, 4, stride=2, pad=1, nobias=True),
            dc4=L.Deconvolution2D(256, 256, 3, stride=1, pad=1, nobias=True),
            dc5=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, nobias=True),
            dc6=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, nobias=True),
            dc7=L.Deconvolution2D(64, 3, 4, stride=2, pad=1, nobias=True),
            bn_dc1=L.BatchNormalization(256),
            bn_dc2=L.BatchNormalization(256),
            bn_dc3=L.BatchNormalization(256),
            bn_dc4=L.BatchNormalization(256),
            bn_dc5=L.BatchNormalization(128),
            bn_dc6=L.BatchNormalization(64),
        )
        self.width=width
        self.height=height
        

    def __call__(self, noise, tag):
        noise_input = self.l1(noise)
        noise_input = F.relu(noise_input)
        tag_input = self.l2(tag)
        tag_input = F.relu(tag_input)
        merge1=F.concat((noise_input,tag_input), axis=1)
        
        h = self.l3(merge1)
        #h = self.bn_l3(h)
        h = F.relu(h)
        h = self.l4(h)
        #h = self.bn_l4(h)
        h = F.relu(h)

        h = F.reshape(h,(-1,128,int(self.height/32),int(self.width/32)))

        h = F.relu(self.bn_dc1(self.dc1(h)))
        h = F.relu(self.bn_dc2(self.dc2(h)))  
        h = F.relu(self.bn_dc3(self.dc3(h)))
        h = F.relu(self.bn_dc4(self.dc4(h)))
        h = F.relu(self.bn_dc5(self.dc5(h)))
        h = F.relu(self.bn_dc6(self.dc6(h)))

        h = F.tanh(self.dc7(h))
        
        
        return h
    
class discriminator(chainer.Chain):
    def __init__(self):
        super().__init__(

            c1=L.Convolution2D(None, 64, 4, stride=2, pad=1, nobias=True),
            c2=L.Convolution2D(64, 128, 4, stride=2, pad=1, nobias=True),
            c3=L.Convolution2D(128, 256, 4, stride=2, pad=1, nobias=True),
            c4=L.Convolution2D(256, 512, 4, stride=2, pad=1, nobias=True),
            c5=L.Convolution2D(512, 256, 4, stride=2, pad=1, nobias=True),
            l1=L.Linear(NUMBER_OF_TAG, 256),
            l2=L.Linear(256, 256),
            l3=L.Linear(256, 256),
  
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
            bn4 = L.BatchNormalization(256),
            l4l = L.Linear(None, 1),
        )


    def __call__(self, image, tag):

        print(image.shape)
        h = F.relu(self.c1(image))
        h = F.relu(self.bn1(self.c2(h)))  
        h = F.relu(self.bn2(self.c3(h)))
        h = F.relu(self.bn3(self.c4(h)))
        h = F.relu(self.bn4(self.c5(h)))

        
        h2 = F.relu(self.l1(tag))
        h2 = F.relu(self.l2(h2))
        h2 = F.relu(self.l3(h2))

        merge1=F.concat((F.reshape(h,(-1 ,256*16*16)),h2), axis=1)
        
        l = self.l4l(merge1)
        
        return l


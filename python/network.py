import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

NUMBER_OF_TAG=1539

class g_block(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super().__init__(
            dc1=L.Deconvolution2D(in_dim, out_dim, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(out_dim, out_dim, 3, stride=1, pad=1),
            bn_dc1=L.BatchNormalization(out_dim),
            bn_dc2=L.BatchNormalization(out_dim),
        )
    def __call__(self,x):
        h = F.leaky_relu(self.bn_dc1(self.dc1(x)))
        return F.leaky_relu(self.bn_dc2(self.dc2(h)))

class generator(chainer.Chain):
    def __init__(self, width, height, z_size):
        super().__init__(
            l1=L.Linear(z_size, 100),
            l2=L.Linear(NUMBER_OF_TAG, 100),
            l3=L.Linear(200, 100),
            l4 = L.Linear(100, int(width/32*height/32*256)),
            
            #bn_l3=L.BatchNormalization(100),
            #bn_l4=L.BatchNormalization(100),
            b0=g_block(256,256),
            b1=g_block(256,256),
            b2=g_block(256,256),
            b3=g_block(256,256),
            b4=g_block(256,256),

            to_RGB=L.Convolution2D(None, 3, 1, stride=1, pad=0),

        )
        self.width=width
        self.height=height
        

    def __call__(self, noise, tag,depth,alpha):
        noise_input = self.l1(noise)
        noise_input = F.leaky_relu(noise_input)
        tag_input = self.l2(tag)
        tag_input = F.leaky_relu(tag_input)
        merge1=F.concat((noise_input,tag_input), axis=1)
        
        h = self.l3(merge1)
        #h = self.bn_l3(h)
        h = F.leaky_relu(h)
        h = self.l4(h)
        #h = self.bn_l4(h)
        h = F.leaky_relu(h)

        h = F.reshape(h,(-1,256,int(self.height/32),int(self.width/32)))

        for i in range(depth-1):
            h = getattr(self, "b%d" % i)(h)
            print("b%d" % i)
        if 0<depth:
            h2 = getattr(self, "b%d" % (depth-1))(h)
            print("b%d" % (depth-1))
            h=F.unpooling_2d(h, 2, 2, 0, outsize=(2*h.shape[2], 2*h.shape[3]))
            h=h*(1.0-alpha)+h2*alpha
        '''
        h=self.b0(h)
        h=self.b1(h)
        h=self.b2(h)
        h=self.b3(h)
        h=self.b4(h)
        '''
        
        
        h = F.tanh(self.to_RGB(h))

        
        return h
    
class d_block(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super().__init__(
            c1=L.Convolution2D(in_dim, out_dim, 3, stride=1, pad=1),
            c2=L.Convolution2D(out_dim, out_dim, 3, stride=1, pad=1),
            bn1 = L.BatchNormalization(out_dim),
            bn2 = L.BatchNormalization(out_dim),
            
        )
    def __call__(self,x,bn=True):
        
        if bn:
            h=F.leaky_relu(self.bn1(self.c1(x)))
            h=F.leaky_relu(self.bn2(self.c2(h)))
            
        else:
            h=F.leaky_relu(self.c1(x))
            h=F.leaky_relu(self.c2(h))  
            
        h=F.average_pooling_2d(h, 2, 2)   
        return h

class std(chainer.Chain):
    def __init__(self):
        super().__init__(
        )
    def __call__(self,x):
        mean=F.mean(x,axis=0)
        dev=x-F.broadcast_to(mean, x.shape)
        devdev=dev*dev
        var=F.mean(devdev)*10
        new_channel = F.broadcast_to(var, (x.shape[0], 1, x.shape[2], x.shape[3]))
        h = F.concat((x, new_channel), axis=1)
        #print(var.data)
        return h


    
class discriminator(chainer.Chain):
    def __init__(self):
        super().__init__(
            from_RGB=L.Convolution2D(3, 256, 1, stride=1, pad=0),    
            
            b0=d_block(257,256),
            b1=d_block(257,256),
            b2=d_block(257,256),
            b3=d_block(257,256),
            b4=d_block(257,256),
            
            l1=L.Linear(NUMBER_OF_TAG, 256),
            #l2=L.Linear(256, 256),
            #l3=L.Linear(256, 256),
            
            #l1=L.Linear(None, NUMBER_OF_TAG),
            #c1=L.Convolution2D(256, 3, 3, stride=2, pad=0, nobias=True),
  
            l4l = L.Linear(None, 1),
            std=std()
        )
        


    def __call__(self, image, tag,depth,alpha):
        h=F.leaky_relu(self.from_RGB(image))
        h2 = F.leaky_relu(self.l1(tag))
        h2 = F.reshape(h2,(h2.shape[0], 1, 16, 16))
        h2=F.unpooling_2d(h2, 2**depth, 2**depth, 0,outsize=(h.shape[2],h.shape[3])) 
        
        #print(h.shape)
        #h2=F.broadcast_to(h2, (h2.shape[0], 1, h.shape[2], h.shape[3]))
        
        #print(h2.shape)
        h=F.concat((h,h2), axis=1)
        #print(h.shape)
        if 0<depth:
            h3 = F.average_pooling_2d(h, 2, 2)
            h=getattr(self, "b%d" % (5-depth))(h,False)
            h2=F.average_pooling_2d(h2, 2, 2)
            h=F.concat((h,h2), axis=1)
            #print(h.shape)
            h=h*alpha+h3*(1-alpha)
            
        
        for i in range(depth-1):
            h = getattr(self, "b%d" % (5-depth+i+1))(h)
            h2=F.average_pooling_2d(h2, 2, 2)
            h=F.concat((h,h2), axis=1)
        
        #print(h.shape)
        h=self.std(h)
        
        '''
        h = self.b0(h,False)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        '''

        #merge1=F.concat((F.reshape(h,(image.shape[0] ,-1)),h2), axis=1)
        
        #print(h.shape)
        
        l = self.l4l(h)

        #l = self.l4l(h)
        return l


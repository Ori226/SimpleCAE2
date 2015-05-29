# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

#from  import activations, initializations
from keras.utils.theano_utils import shared_zeros
from keras.layers.core import Layer

class ReverseMaxPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(ReverseMaxPooling2D,self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)

        up_pool_out = theano.tensor.extra_ops.repeat(X, self.poolsize[0], axis = 2).repeat(self.poolsize[1], axis = 3)

        output = up_pool_out
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize,
            "ignore_border":self.ignore_border}



if __name__ == "__main__":
    print 'hello'

    import numpy as np
    temp_input = np.asarray([[1,2],[3,4]], dtype= theano.config.floatX)
    new_layer =  ReverseMaxPooling2D()
    after_revpool =  new_layer.get_output(temp_input)

    print after_revpool





    #now test it:

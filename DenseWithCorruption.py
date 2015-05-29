from __future__ import absolute_import, division
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Layer
from keras.optimizers import SGD, RMSprop
import numpy as np


from utils import tile_raster_images as show_row_vectors

import skimage
import sklearn
from sklearn import preprocessing,cross_validation


# -*- coding: utf-8 -*-
import os,sys,pdb
import imsave2
import theano
import theano.tensor as T

from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros, floatX
from keras.utils.generic_utils import make_tuple

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
from scipy import io as sp_io





class DenseWithCorruption(Layer):
    '''
        Just your regular fully connected NN layer.
    '''
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, corruption_level=0.0):

        super(DenseWithCorruption,self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        self.regularizers = [W_regularizer, b_regularizer]
        self.constraints = [W_constraint, b_constraint]
        
        numpy_rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.corruption_level = corruption_level
        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        tilde_x = self.get_corrupted_input(X, self.corruption_level)
        output = self.activation(T.dot(tilde_x, self.W) + self.b)
        return output

    def get_corrupted_input(self, input, corruption_level):        
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}
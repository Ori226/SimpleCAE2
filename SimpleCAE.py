import theano

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ReverseMaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils, generic_utils
import numpy as np
import random
import pylab
from PIL import Image


import keras.datasets.cifar10 
#use the cifar10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data(test_split=0.1, seed=113)

number_of_class = 1000
X_train = X_train[0:number_of_class,:,:,:]
y_train = y_train[0:number_of_class,:]


#
nb_classes = 10


model = Sequential()



in_size = 32*32
im_w = 32
im_h = 32
in_filter_w = 3
in_filter_h = 3

model.add(Convolution2D(10, 3, in_filter_h, in_filter_w, border_mode='full')) 
im_w = im_w + in_filter_w - 1
im_h = im_h + in_filter_h -1 


pool_w = 2
pool_h = 2
model.add(Convolution2D(3, 10, in_filter_h, in_filter_w, border_mode='valid')) 
model.add(MaxPooling2D(poolsize=(pool_h, pool_w)))
model.add(Activation('relu'))
model.add(DenseWithCorruption('relu'))


model.add(Dropout(0.2))
model.add(ReverseMaxPooling2D(poolsize=(pool_h, pool_w))) 


model.add(Flatten())
temp = X_train.flatten(1)

valid_set = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))



rmsprop = RMSprop(lr=0.0001, rho=0.5, epsilon=1e-6)
model.compile(loss='mean_squared_error', optimizer=rmsprop)


model.fit(X_train, valid_set, nb_epoch=100,batch_size=128)


reconstruction_results =  model.predict_proba(X_train)
reconstruction_results = reconstruction_results.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])




index1 = random.randint(0,number_of_class)
index2 = random.randint(0,number_of_class)
index3 = random.randint(0,number_of_class)
index4 = random.randint(0,number_of_class)


pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(X_train[index1,0,:,:]);
pylab.gray();
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(reconstruction_results[index1,0,:,:])



# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:

pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(X_train[index2,0,:,:])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(reconstruction_results[index2,0,:,:])


pylab.show()
#print('Hello World')

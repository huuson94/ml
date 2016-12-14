#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import skimage.transform 
from scipy import misc

MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
def preprocess(img):
    print(type(img))
    print(img.shape)
    # img is (channels, height, width), values are 0-255
    img = img[::-1]  # switch to BGR
    img -= MEAN_VALUE
    return img

def load_data_file(class_file):
    X_data = []
    y_data = []
    res_root = "/home/oanhnt/sonnh/src/ml/VOCdevkit/VOC2012/JPEGImages/"
    with open(class_file) as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        filename, class_name, class_no = line.split()
        imarr = misc.imresize(misc.imread(res_root+ filename), (224, 224, 3)).astype(np.float32) 
        #reshape to (3, 224, 224)
        imarr_224 = np.swapaxes(np.swapaxes(imarr,0,1),1,2)    
        reshape_img = np.swapaxes(np.swapaxes(imarr_224,1,2),0,1).reshape((3,224,224))
        X_data.append(preprocess(reshape_img))
        y_data.append(int(class_no))
    return np.array(X_data), np.array(y_data)

def load_dataset():
    X_train, y_train = load_data_file("train_image_class.txt")
    X_val, y_val = load_data_file("valid_image_class.txt")
    return X_train, y_train, X_val, y_val
def build_vgg(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 3, 224, 224), input_var=input_var, name="input")
    network = lasagne.layers.Conv2DLayer(network, num_filters = 64, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 64, filter_size=(3,3), pad = 1)
    network= lasagne.layers.MaxPool2DLayer(network, pool_size = 2)

    network = lasagne.layers.Conv2DLayer(network, num_filters = 128, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 128, filter_size=(3,3), pad = 1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2)

    network = lasagne.layers.Conv2DLayer(network, num_filters = 256, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 256, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 256, filter_size=(3,3), pad = 1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2)

    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2)

    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5), 
            num_units = 4096, 
            nonlinearity=lasagne.nonlinearities.rectify
            ) 
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5), 
            num_units = 4096, 
            nonlinearity=lasagne.nonlinearities.rectify
            ) 
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5), 
            num_units = 1000, 
            nonlinearity=lasagne.nonlinearities.softmax 
            ) 

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def main():
    num_epochs = 5
    X_train, y_train, X_val, y_val = load_dataset()
    print("Building net...")
    print("Build done")

    print("Create train variables")
    input_var = T.tensor4('inputs')
    target_var = T.lvector('targets')
    network = build_vgg(input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    print("Done creating train variables")
    
    print("Starting traninig...")

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

    # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


if __name__ == '__main__':
    main()


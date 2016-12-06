#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

def build_vgg():
    input_layer = lasagne.layers.InputLayer(shape=(None, 3, 224, 224), input_var=None, name="input")
    conv11 = lasagne.layers.Conv2DLayer(input_layer, num_filters = 64, filter_size=(3,3), pad = 1)
    conv12 = lasagne.layers.Conv2DLayer(conv11, num_filters = 64, filter_size=(3,3), pad = 1)
    pool1 = lasagne.layers.MaxPool2DLayer(conv12, pool_size = 2)

    conv21 = lasagne.layers.Conv2DLayer(pool1, num_filters = 128, filter_size=(3,3), pad = 1)
    conv22 = lasagne.layers.Conv2DLayer(conv21, num_filters = 128, filter_size=(3,3), pad = 1)
    pool2 = lasagne.layers.MaxPool2DLayer(conv22, pool_size = 2)

    conv31 = lasagne.layers.Conv2DLayer(pool2, num_filters = 256, filter_size=(3,3), pad = 1)
    conv32 = lasagne.layers.Conv2DLayer(conv31, num_filters = 256, filter_size=(3,3), pad = 1)
    conv33 = lasagne.layers.Conv2DLayer(conv32, num_filters = 256, filter_size=(3,3), pad = 1)
    pool3 = lasagne.layers.MaxPool2DLayer(conv33, pool_size = 2)

    conv41 = lasagne.layers.Conv2DLayer(pool3, num_filters = 512, filter_size=(3,3), pad = 1)
    conv42 = lasagne.layers.Conv2DLayer(conv41, num_filters = 512, filter_size=(3,3), pad = 1)
    conv43 = lasagne.layers.Conv2DLayer(conv42, num_filters = 512, filter_size=(3,3), pad = 1)
    pool4 = lasagne.layers.MaxPool2DLayer(conv43, pool_size = 2)

    conv51 = lasagne.layers.Conv2DLayer(pool4, num_filters = 512, filter_size=(3,3), pad = 1)
    conv52 = lasagne.layers.Conv2DLayer(conv51, num_filters = 512, filter_size=(3,3), pad = 1)
    conv53 = lasagne.layers.Conv2DLayer(conv52, num_filters = 512, filter_size=(3,3), pad = 1)
    pool5 = lasagne.layers.MaxPool2DLayer(conv53, pool_size = 2)

    fc6 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool5, p=.5), 
            num_units = 4096, 
            nonlinearity=lasagne.nonlinearities.softmax 
            ) 
    fc7 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool5, p=.5), 
            num_units = 4096, 
            nonlinearity=lasagne.nonlinearities.softmax 
            ) 
    fc8 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(pool5, p=.5), 
            num_units = 1000, 
            nonlinearity=lasagne.nonlinearities.softmax 
            ) 

    return fc8


def main():
    network = build_vgg()

if __name__ == '__main__':
    main()


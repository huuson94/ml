import os
import matplotlib.pyplot as plt
import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params
from lasagne.layers import set_all_param_values
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import PrintLayerInfo
import pickle
from scipy import misc
#read trainned weights

layers0 = [        
        (InputLayer, {'shape': (None, 3, 224, 224)}),
        #conv11
        (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad' : 1}),
        #conv12
        (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad' : 1}),
        #pool1
        (MaxPool2DLayer, {'pool_size': 2}),
        #conv21
        (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad' : 1}),
        #conv22
        (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad' : 1}),
        #pool2
        (MaxPool2DLayer, {'pool_size': 2}),
        #conv31
        (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad' : 1}),
        #conv32
        (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad' : 1}),
        #conv33
        (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad' : 1}),
        #pool3
        (MaxPool2DLayer, {'pool_size': 2}),
        #conv41
        (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad' : 1}),
        #conv42
        (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad' : 1}),
        #conv43
        (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad' : 1}),
        #pool4
        (MaxPool2DLayer, {'pool_size': 2}),
        #conv51
        (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad' : 1}),
        #conv52
        (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad' : 1}),
        #conv53
        (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad' : 1}),
        #pool5
        (MaxPool2DLayer, {'pool_size': 2}),
        #fc6
        (DenseLayer, {'num_units': 4096, 'nonlinearity': softmax}),
        (DropoutLayer,{}),
         #fc7
        (DenseLayer, {'num_units': 4096, 'nonlinearity': softmax}),
        (DropoutLayer,{}),
        #fc8
        (DenseLayer, {'num_units': 1000, 'nonlinearity': softmax}),
        ]
net0 = NeuralNet(
        layers=layers0,
        max_epochs=0,
        update=adam,
        update_learning_rate=0.0002,
        objective_l2=0.0025,
        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
        )
net0.load_params_from('vgg16.pkl')
net0.initialize()
layer_info = PrintLayerInfo()
print(layer_info._get_greeting(net0))

data = misc.imread('judo_test.jpg')
layer_info, legend = layer_info._get_layer_info_conv(net0)
print(layer_info)
print("Predicted: %s " %str(net0.predict(data.reshape(-1, 3, 224, 224))))









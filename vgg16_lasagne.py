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

from training_process_file import read_origin_params
from training_process_file import read_params
from training_process_file import save_params
from training_process_file import save_history

from visualize import plot_loss
from visualize import plot_conv1_weights

from modify_data import modify_sample

MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype="int32")   # BGR
DEV_PATH = '/home/hs/workspace/python/ml/101_ObjectCategories'
SERVER_PATH = '/home/oanhnt/sonnh/src/ml/101_ObjectCategories'
TRAIN_VALID_RATIO = 0.7
snapshot_root = 'snapshot_models/'

#SAMPLE_NUMBER = 9000

classes_name = []

def preprocess(img):
    # img is (channels, height, width), values are 0-255
    img = img[::-1, :, :]  # switch to BGR
    img[0] -= MEAN_VALUE[0]
    img[1] -= MEAN_VALUE[1]
    img[2] -= MEAN_VALUE[2]
    return img

def load_data_folder():
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    res_root = DEV_PATH
    
    dirs = os.listdir(res_root)

    for dir in dirs:
        class_name = dir
        classes_name.append(class_name)
        
        files = os.listdir(res_root + "/" + dir)
        train_sample_count_limit = len(files) * TRAIN_VALID_RATIO - 1

        for index, file in enumerate(files):
            if(index < train_sample_count_limit):
                X_train.append(load_data_from_file(res_root + "/" + dir + "/" + file))
                y_train.append(classes_name.index(class_name))
            else:
                X_val.append(load_data_from_file(res_root + "/" + dir + "/" + file))
                y_val.append(classes_name.index(class_name))

    return np.array(X_train, dtype="float32"), np.array(y_train, dtype="int32"), np.array(X_val, dtype="float32"), np.array(y_val, dtype="int32")
count = 0
def load_data_from_file(file_path, height_crop=224, width_crop=224):
    img = misc.imread(file_path)
    if(len(img.shape) == 2):
        height, width= img.shape
        new_arr = np.zeros((height, width, 3))
        new_arr[:,:,0] = img
        new_arr[:,:,1] = img
        new_arr[:,:,2] = img
        img = new_arr
    
    height, width, color = img.shape
    min_size = height if height <= width else width
    #center crop 
    startX = width//2 - min_size//2
    startY = height//2 - min_size//2
    cropped_img = img[startY:startY+min_size, startX:startX+min_size,:]
    img = misc.imresize(img, (height_crop, width_crop), interp='nearest')

    #reshape to (3, 224, 224)
    img_021= np.swapaxes(np.swapaxes(cropped_img,0,1),1,2)    
    reshape_img = np.swapaxes(np.swapaxes(img_021,0,1),1,2)

    return preprocess(reshape_img)

def load_dataset():
    X_train, y_train, X_val, y_val = load_data_folder()
    return X_train, y_train, X_val, y_val

def build_vgg(input_var):
    origin_params = read_origin_params();
    training_params = read_params(snapshot_root + 'vgg16_lasagne170116085327_params')
    network = lasagne.layers.InputLayer(shape=(None, 3, 224, 224), input_var=input_var, name="input")
    network = lasagne.layers.Conv2DLayer(network, num_filters = 64, filter_size=(3,3), pad = 1, name="conv1_1", W = origin_params[0], b = origin_params[1])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 64, filter_size=(3,3), pad = 1, name="conv1_2", W = origin_params[2], b = origin_params[3])
    network= lasagne.layers.MaxPool2DLayer(network, pool_size = 2, stride=2, name="pool1")

    network = lasagne.layers.Conv2DLayer(network, num_filters = 128, filter_size=(3,3), pad = 1, W = origin_params[4], b = origin_params[5])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 128, filter_size=(3,3), pad = 1, W = origin_params[6], b = origin_params[7])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2, stride=2, name="pool2")

    network = lasagne.layers.Conv2DLayer(network, num_filters = 256, filter_size=(3,3), pad = 1, W = origin_params[8], b = origin_params[9])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 256, filter_size=(3,3), pad = 1, W = origin_params[10], b = origin_params[11])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 256, filter_size=(3,3), pad = 1, W = origin_params[12], b = origin_params[13])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2, stride=2, name="pool3")

    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1, W = origin_params[14], b = origin_params[15])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1, W = origin_params[16], b = origin_params[17])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1, W = origin_params[18], b = origin_params[19])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2, stride=2, name="pool4")

    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1, W = origin_params[20], b = origin_params[21])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1, W = origin_params[22], b = origin_params[23])
    network = lasagne.layers.Conv2DLayer(network, num_filters = 512, filter_size=(3,3), pad = 1, W = origin_params[24], b = origin_params[25])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = 2, stride=2, name="pool5")

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5), 
            num_units = 4096, 
            nonlinearity=lasagne.nonlinearities.rectify, 
            name = "fc6",
            W = origin_params[26],
            b = origin_params[27]
            ) 
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5), 
            num_units = 4096, 
            nonlinearity=lasagne.nonlinearities.rectify,
            name = "fc7",
            W = origin_params[28],
            b = origin_params[29]
            ) 
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5), 
            num_units = 1000, 
            nonlinearity=lasagne.nonlinearities.softmax,
            W = training_params[0],
            b = training_params[1]
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


def modify(inputs, number_sample):
    rs = []
    for input in inputs:
        rs.append(modify_sample(input,number_sample = number_sample))
    return rs

def main():
    num_epochs = 500
    print("Load dataset...") 
    X_train, y_train, X_val, y_val = load_dataset()
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    print("Building net...")
    #network = build_vgg(input_var)
    print("Create train variables")


    time_stamp=time.strftime("%y%m%d%H%M%S", time.localtime()) 
    snapshot_name = 'vgg16_lasagne'+ time_stamp


    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    params = params[-2:]


    learning_rate_init = 1e-3
    learning_rate = theano.shared(np.array(learning_rate_init, dtype=theano.config.floatX))
    updates = lasagne.updates.adam(
            loss, params, learning_rate=learning_rate_init)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    print("Starting tranining...")
    training_history = {}
    training_history['iter_training_loss'] = []
    training_history['iter_validation_loss'] = []
    training_history['training_loss'] = []
    training_history['validation_loss'] = []
    training_history['learning_rate'] = []
    
    iter_now = 0
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        X_train = modify(X_train, (epoch + 1) * 10)
        print(np.median(X_train))
        exit(0)
        for batch in iterate_minibatches(X_train, y_train, 200, shuffle=True):
            inputs, targets = batch
            #train_err += train_fn(inputs, targets)
            print("Train batch {} took {:.3f}s, loss:{:.6f}".format(
                 train_batches + 1, time.time() - start_time, train_err / (train_batches + 1)))
            train_batches += 1
            iter_now += 1
    # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 200, shuffle=False):
            inputs, targets = batch
            #err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            print("Valid batch {} took {:.3f}s, loss:{:.6f}".format(
                 val_batches+ 1, time.time() - start_time, val_err/(val_batches +1)))
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        #plot loss
        training_history['iter_training_loss'].append(train_err/train_batches)
        training_history['iter_validation_loss'].append(val_err / val_batches)
        training_history['learning_rate'].append(learning_rate.get_value())
        #if(epoch != 0 and (epoch % 10) == 0):
        if(epoch != 0 and (epoch % 10) == 0):
            snapshot_path_string = snapshot_root+snapshot_name+'_'+str(epoch)+"_"+str(iter_now+1)
            print("Save param")
            save_params(params, snapshot_path_string+"_"+"params")
            print("Creating snapshot")
            plot_loss(training_history, snapshot_path_string+'_loss.png')
            print("Creating weight visualize")
            plot_conv1_weights(lasagne.layers.get_all_layers(network)[1], snapshot_path_string + '_conv1weights.png')
            print("Save training history")

    print("Trainning done")
    save_params(params)


if __name__ == '__main__':
    main()


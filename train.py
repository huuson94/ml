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

from training_process_file import save_params
from training_process_file import save_history

from visualize import plot_loss
from visualize import plot_conv1_weights

from modify_data import modify_sample

from build_vgg16 import build_vgg16
from build_resnet50 import build_resnet50
from build_lnet5 import build_lnet5
#from build_googlenet import build_googlenet

MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype="float32")   # BGR
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

def load_data_folder(height_crop, width_crop):
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    res_root = SERVER_PATH
    
    dirs = os.listdir(res_root)

    for dir in dirs:
        class_name = dir
        classes_name.append(class_name)
        
        files = os.listdir(res_root + "/" + dir)
        train_sample_count_limit = len(files) * TRAIN_VALID_RATIO - 1

        for index, file in enumerate(files):
            if(index < train_sample_count_limit):
                X_train.append(load_data_from_file(res_root + "/" + dir + "/" + file,height_crop=height_crop, width_crop=width_crop))
                y_train.append(classes_name.index(class_name))
            else:
                X_val.append(load_data_from_file(res_root + "/" + dir + "/" + file,height_crop=height_crop, width_crop=width_crop))
                y_val.append(classes_name.index(class_name))

    return np.array(X_train, dtype="float32"), np.array(y_train, dtype="int32"), np.array(X_val, dtype="float32"), np.array(y_val, dtype="int32")
    #return np.array(X_train[0:100], dtype="float32"), np.array(y_train[0:100], dtype="int32"), np.array(X_val[0:100], dtype="float32"), np.array(y_val[0:100], dtype="int32")

def load_data_from_file(file_path, height_crop, width_crop):
    img = misc.imread(file_path).astype('float32')
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
    img = misc.imresize(img, (height_crop, width_crop), interp='nearest').astype('float32')

    #reshape to (3, 224, 224)
    img_021= np.swapaxes(np.swapaxes(img,0,1),1,2)    
    reshape_img = np.swapaxes(np.swapaxes(img_021,0,1),1,2)

    return preprocess(reshape_img)

def load_dataset(height_crop=224, width_crop=224):
    print("Load dataset...") 
    X_train, y_train, X_val, y_val = load_data_folder(height_crop=height_crop, width_crop=width_crop)
    return X_train, y_train, X_val, y_val


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
        modified_input = modify_sample(input,number_sample = number_sample)
        rs.append(modified_input)
    return np.array(rs, dtype="float32")

def main(model='vgg16', num_epochs=100):
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    print("Building net...")
    if(model == 'vgg16'):
        network = build_vgg16(input_var)
    if(model == 'resnet50'):
        network = build_resnet50(input_var)
        network = network['prob']
        X_train, y_train, X_val, y_val = load_dataset(224, 224)
    if(model == 'googlenet'):
        pass
        #network = build_googlenet(input_var)
    if(model == 'lnet5'):
        X_train, y_train, X_val, y_val = load_dataset(28, 28)
        network = build_lnet5(input_var)
    print("Create train variables")

    time_stamp=time.strftime("%y%m%d%H%M%S", time.localtime()) 
    snapshot_name = model + time_stamp


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
    batch_size = 200
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #print(X_train.shape)
        X_train = modify(X_train, (epoch + 1) * 10)
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            print("Train batch {} took {:.3f}s, loss:{:.6f}".format(
                 train_batches + 1, time.time() - start_time, train_err / (train_batches + 1)))
            train_batches += 1
            iter_now += 1
    # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size , shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
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
    kwargs = {}
    if len(sys.argv) > 1:
       kwargs['model'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['num_epochs'] = int(sys.argv[2])
    main(**kwargs)


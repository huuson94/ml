import matplotlib
# matplotlib asked for Xserver backend by default, causing unwanted error
# Force matplotlib not to use any Xwindows backend.
matplotlib.use('Agg')

from itertools import product

from lasagne.layers import get_output
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T


def plot_loss(training_history, filename=""):
    train_loss = training_history['iter_training_loss']
    valid_loss = training_history['iter_validation_loss']
    plt.figure() # !
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss (epoch)')
    plt.xlabel('iter')
    plt.ylabel('loss')
    ymax = max(training_history['iter_training_loss'] + training_history['iter_validation_loss']) * 1.2
    plt.ylim(0,ymax) # !
    plt.legend(loc='best')

    if len(filename)>0:
        plt.savefig(filename) # !


def bc01_to_b01c(X):
    return np.swapaxes(np.swapaxes(X,1,2),2,3)

def color_convert(X):
    height, width, color = X.shape
    if(color == 1):
        new_image = np.zeros((height, width, 3))
        new_image[:,:,0] = X[:,:,0]
        new_image[:,:,1] = X[:,:,0]
        new_image[:,:,2] = X[:,:,0]
        return new_image
    return X

def plot_conv1_weights(layer, filename="", figsize=(6,6)):
    
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    W = bc01_to_b01c(W) # !
    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        image = color_convert(W[i,:])#black white image
        axes[r, c].imshow(image, interpolation='nearest')
        
    if len(filename)>0:
        plt.savefig(filename) # !
   



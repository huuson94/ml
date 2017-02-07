import lasagne
from training_process_file import read_origin_params
from training_process_file import read_params
#from train import snapshot_root
snapshot_root = "snapshot_models/"

def build_vgg16(input_var):
    origin_params = read_origin_params('vgg16');
    training_params = read_params(snapshot_root+'vgg16_lasagne170116085327_params')
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



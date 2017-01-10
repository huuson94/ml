import pprint, pickle

def read_params():

    pkl_file = open('vgg16.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    #pprint.pprint(data1['param values'][0].shape)
    pkl_file.close()
    return data1['param values']

def save_params(params):
    trainned_params = open('vgg16_trainning.pkl', 'wb')
    pickle.dump(params, trainned_params)
    trainned_params.close()


#read_params()

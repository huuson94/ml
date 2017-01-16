import pprint, pickle

def read_origin_params():
    pkl_file = open("vgg16.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data['param values']

def read_params(training_params_path):
    pkl_file = open(training_params_path + ".pkl", 'rb')
    training_params = pickle.load(pkl_file)
    pkl_file.close()
    return training_params

def save_params(params, training_params_path):
    training_params = open(training_params_path+".pkl", 'wb')
    pickle.dump(params, training_params)
    training_params.close()

def save_history(training_history, training_history_path):
    training_history = open(training_history_path+".pkl", 'wb')
    pickle.dump(training_history, training_history_path)
    training_history.close


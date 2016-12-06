import pprint, pickle

pkl_file = open('vgg16.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1['param values'][6].shape)

pkl_file.close()

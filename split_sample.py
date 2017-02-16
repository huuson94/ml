import sys
import os
import random

DEV_PATH = '/home/hs/workspace/python/ml/101_ObjectCategories'
SERVER_PATH = '/home/oanhnt/sonnh/src/ml/101_ObjectCategories'
RES_ROOT = DEV_PATH
TRAIN_VALID_SAMPLES = 30
TRAIN_VALID_RATIO = 0.7
TRAIN_VALID_OUTPUT_PATH = RES_ROOT + '/' + 'train'
TEST_OUTPUT_PATH = RES_ROOT + '/' + 'test'

def read_dir():
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    train_output_file = open('train.txt', 'wb')
    valid_output_file = open('valid.txt', 'wb')
    test_output_file = open('test.txt', 'wb')

    dirs = os.listdir(RES_ROOT)
    classes_name = []

    for dir in dirs:
        class_name = dir
        classes_name.append(class_name)
        files = os.listdir(RES_ROOT + "/" + dir)
        random.shuffle(files)
        train_sample_count_limit = len(files) * TRAIN_VALID_RATIO - 1

        for index, file in enumerate(files):
            if(index < TRAIN_VALID_SAMPLES):
                if( (index + 1)  < TRAIN_VALID_SAMPLES * TRAIN_VALID_RATIO):
                    train_output_file.write(str(index) +'\t' + dir + '/' + file + '\t' + str(classes_name.index(class_name)) + '\n')
                else:
                    valid_output_file.write(str(index) +'\t' + dir + '/' + file + '\t' + str(classes_name.index(class_name)) + '\n')
            else:
                test_output_file.write(str(index) + '\t' + dir + '/' + file + '\t' + str(classes_name.index(class_name)) + '\n')

    train_output_file.close()
    test_output_file.close()

read_dir()

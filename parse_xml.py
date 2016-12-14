import os, numpy
import xml.etree.ElementTree as parser

CLASSES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "dining" "table", "dog", "horse", "motorbike", "person",
        "potted" "plant", "sheep", "train", "tvmonitor", "bird", "sofa"]

START_TRAIN_NUMBER = 0 
END_TRAIN_NUMBER = 99 
START_VALID_NUMBER = 100
END_VALID_NUMBER = 199

def parse_xml(file_path):
    root_node = parser.parse(file_path).getroot()
    class_name = root_node.find("object").find("name").text
    return root_node.find("filename").text + "\t" + class_name + "\t"+ str(CLASSES.index(class_name)) + "\n"

res_root = '/home/hs/workspace/python/ml/train_val_data/VOC2012/Annotations/'
allfiles = os.listdir(res_root)
allfiles.sort()

imlist = [filename for filename in allfiles if filename[-4:] in [".xml"]]
train_result_file = open("train_image_class.txt", "wb")
for im in imlist[START_TRAIN_NUMBER:END_TRAIN_NUMBER]:
    train_result_file.write(parse_xml(res_root + im))
train_result_file.close()

valid_result_file = open("valid_image_class.txt", "wb")
for im in imlist[START_VALID_NUMBER:END_VALID_NUMBER]:
    valid_result_file.write(parse_xml(res_root + im))
valid_result_file.close()


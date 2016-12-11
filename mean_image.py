import os, numpy, PIL
from scipy import misc
import skimage.transform

res_root = '/home/hs/workspace/python/ml/train_val_data/VOC2012/JPEGImages/'
# Access all PNG files in directory
allfiles=os.listdir(res_root)
allfiles.sort()

imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

# Assuming all images are the same size, get dimensions of first image
#h, w, d  =misc.imread(res_root+ imlist[0]).shape
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((227, 227, 3),numpy.float32)

# Build up average pixel intensities, casting each image as an array of floats
#loop_max = numpy.float32(1000)
for index, im in enumerate(imlist):
    imarr = misc.imresize(misc.imread(res_root+ im), (227, 227, 3)).astype(numpy.float32)
    arr=arr+imarr/N
    #if index == loop_max - 1:
    #    break

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)
misc.toimage(arr).show()
arr_227 = numpy.swapaxes(numpy.swapaxes(arr,0,1),1,2)    
arr = numpy.swapaxes(numpy.swapaxes(arr_227,1,2),0,1).reshape((1,3,227,227))
#print(arr.shape)
numpy.savez('mean_image.npz', arr)

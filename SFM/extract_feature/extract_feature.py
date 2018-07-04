def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)
    
    
import logging
import cv2
import numpy as np
import os
import time
import get_image
import cPickle as pickle
import multiprocessing 
from multiprocessing import Pool


path_logging = '/home/indshine-2/Downloads/Dimension/Dimension/logging/';
output_location = '/home/indshine-2/Downloads/Dimension/output/SFM/extract_feature/';

path_image = '/home/indshine-2/Downloads/Dimension/Data/City/'
list_image = get_image.list_image(path_image,path_logging);
num_feature = 40000;

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s',
                    filename= path_logging + '/sift.log');
                    #,level=logging.info);
               
def extract_feature(image):
    
    im = cv2.imread(image);
    _gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _start_time = time.time();
    sift = cv2.xfeatures2d_SIFT.create(nfeatures = num_feature);
    
    kp, des = sift.detectAndCompute(_gray,None);

    
    #Store and Retrieve keypoint features
    temp = pickle_keypoints(kp, des)
    output = output_location + os.path.basename(image) + ".mansift";
    pickle.dump(temp, open(output, "wb"))
    
    # Printing and logging info
    _end_time = time.time()
    _time_taken = _end_time - _start_time
    logging.info('Image %s processed, took : %s seconds'%(os.path.basename(image),str(_time_taken)))

    print('finished processing %s and took time %s seconds'%(image,_time_taken) )
    return output

def pool_handler():
    p = Pool(processes=4,maxtasksperchild=1000);
    p.map(extract_feature, list_image)
    
def main():
    pool_handler()

main()
#Retrieve Keypoint Features
#keypoints_database = pickle.load( open( "keypoints_database.mansift", "rb" ) )
#kp1, desc1 = unpickle_keypoints(keypoints_database[0])
#kp1, desc1 = unpickle_keypoints(keypoints_database[1])

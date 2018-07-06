""" Extracting features from images, extract_feature.py"""
def checkdir(path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
        
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
    
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")
    
    
import logging
reload(logging)
import cv2
import numpy as np
import os
import time
import get_image
import cPickle as pickle
import multiprocessing 
from multiprocessing import Pool
import sys
import argparse

parser = argparse.ArgumentParser(description='See description below to see all available options')

parser.add_argument('-i','--input',
                    help='Input directory containing images', 
                    default= './',
                    required=True)
                    
parser.add_argument('-o','--output',
                    help='Output directory where all the output will be stored',
                    default = './Output/',
                    required=False)
                    
parser.add_argument('-m','--method',type=int,
                    help='Methods that will be used to calculate '+ 
                    'detectors. Enter an integer value corresponding to options'+
                    ' given below. Available methods are ' + 
                    ' 1 -- SIFT ' +
                    ' 2 -- SURF'  +
                    ' 3 -- ORB'   +
                    ' 4 -- BRISK' +
                    ' 5 -- AKAZE',
                    default = 1,
                    required=False)
                                        

args = parser.parse_args()

path_image = args.input;
output_path = args.output;

checkdir(output_path);
method = args.method;

# Path locations
path_logging = '/home/indshine-2/Downloads/Dimension/Dimension/logging/';
report_logging = '/home/indshine-2/Downloads/Dimension/Dimension/report/';
#output_location = '/home/indshine-2/Downloads/Dimension/output/SFM/extract_feature/';
#path_image = '/home/indshine-2/Downloads/Dimension/Data/City/'


# Checking if path  exists, otherwise will be created
checkdir(path_logging);
checkdir(report_logging);

# Parameters
num_feature = 40000;
count_feature = [];
reload(logging)
logging.basicConfig(format='%(asctime)s %(message)s',
                    filename= path_logging + '/extract_feature.log',
                    level=logging.DEBUG);

# Getting list of images present in given folder.
list_image = get_image.list_image(path_image,path_logging);

                    
if not os.path.exists(os.path.dirname(path_image)):
    logging.fatal('Input directory given was not found')
    sys.exit()
    
overall_time = time.time();
          
def extract_feature(image):
    try:
        im = cv2.imread(image);
        _gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        _start_time = time.time();
        
        if method == 1: # Sift 
            method_ = 'sift/'
            sift_ = cv2.xfeatures2d_SIFT.create()#nfeatures = num_feature); # Predefining number of features
            kp, des = sift_.detectAndCompute(_gray,None);
            count_feature.append(len(kp));
            
        if method ==2: # Surf
            method_ = 'surf/'
            surf_ = cv2.xfeatures2d.SURF_create();
            kp, des = surf_.detectAndCompute(_gray,None);
            count_feature.append(len(kp));
            
        if method ==3: # ORB
            method_ = 'orb/'
            orb_ = cv2.ORB_create()#nfeatures = num_feature) # Predefining number of features
            kp,des = orb_.detectAndCompute(_gray,None);    
            count_feature.append(len(kp));
            
        if method == 4: # Brisk
            method_ = 'brisk/'
            brisk_ = cv2.BRISK_create();
            kp, des = brisk_.detectAndCompute(_gray,None);
            count_feature.append(len(kp));
            
        if method == 5: # AKAZE
            method_ = 'akaze/'
            akaze_ = cv2.AKAZE_create();
            kp, des = akaze_.detectAndCompute(_gray,None);
            count_feature.append(len(kp));
            
#       Store and Retrieve keypoint features
        temp = pickle_keypoints(kp, des)
        output = output_path +method_;
        
#       checking output directory if this exist otherwise will be created
        checkdir(output);
        output = output + os.path.splitext(os.path.basename(image))[0]
        
#       Saving features as numpy compressed array
        np.save(output,temp)
        
#        Printing and logging info
        _end_time = time.time()
        _time_taken = (_end_time - _start_time)/5
        logging.info('Image %s processed, took : %s sec per thread'
        %(os.path.basename(image),str(round(_time_taken,1))))
    
        print('finished processing %s and took %s sec per thread'
        %(os.path.splitext(os.path.basename(image))[0],_time_taken) )
        
    except KeyboardInterrupt:
        pause()
#        sys.exit('KeyboardInterruption, Terminated')
        
#     Returns list of images
    return output

def pool_handler():
    p = Pool(processes=5,maxtasksperchild=1);
    output, count_feature = p.map(extract_feature, list_image)
    print(count_feature)
    p.close()

    
def main():
    try:
        pool_handler()
    except KeyboardInterrupt:
        logging.warning('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')
        
if __name__ == '__main__':
    try:
        main()
        _time_taken_ = str(round(time.time() - overall_time,1))
        logging.info('Total time taken is %s sec'%(_time_taken_))
        
    except KeyboardInterrupt:
        logging.warning('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')
        
#Retrieve Keypoint Features
#keypoints_database = pickle.load( open( "keypoints_database.mansift", "rb" ) )
#kp1, desc1 = unpickle_keypoints(keypoints_database)


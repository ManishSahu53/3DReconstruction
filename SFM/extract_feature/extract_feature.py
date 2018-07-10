""" Extracting features from images, extract_feature.py"""
import logging
reload(logging)
  
def checkdir(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id)#, descriptors[i])     
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
    
def tojson(dictA,file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f,indent=4, separators=(',', ': '), ensure_ascii=False,encoding='utf-8')
        
def create_logger(name):
    """Create a logger

    args: name (str): name of logger

    returns: logger (obj): logging.Logger instance
    """
    logger = multiprocessing.get_logger()
    fmt = logging.Formatter('%(asctime)s - %(name)s -'
                            ' %(levelname)s -%(message)s')
    hdl = logging.FileHandler(name+'.log')
    hdl.setFormatter(fmt)
    logger.addHandler(hdl)
    return logger
    
    
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
import json 
#import logging
#reload(logging)


parser = argparse.ArgumentParser(description='See description below to see all available options')

parser.add_argument('-i','--input',
                    help='Input directory containing images. [Default] current directory',
                    required=True)
                    
parser.add_argument('-o','--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
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
                    ' 5 -- AKAZE' +
                    ' 6 -- STAR+ BRIEF',
                    default = 1,
                    required=False)
                                        

args = parser.parse_args()

path_image = args.input;
output_path = args.output;

checkdir(output_path);
method = args.method;

# Check if method > 5
if method > 6 or method <1:
    sys.exit('Invalid method selected. Only %s methods are available. See help to know how to use them '%(str(5)))

# Path locations
path_logging = './logging/';
#report_logging = './report/';
#output_location = '/home/indshine-2/Downloads/Dimension/output/SFM/extract_feature/';
#path_image = '/home/indshine-2/Downloads/Dimension/Data/test/'

# Setup logging of function 
#logging.basicConfig(format='%(asctime)s %(message)s',
#                    filename= path_logging + '/extract_feature.log',
#                    level=logging.DEBUG);

logname= path_logging + '/extract_feature'
#logger = create_logger(logname)
#logger.info('Starting up logs')

# Checking if path  exists, otherwise will be created
checkdir(path_logging);
#checkdir(report_logging);

# Parameters
num_feature = 40000;
count_feature = [];
append_image = [];
append_time = [];
append_method = [];
threads = 6;

# Getting list of images present in given folder.
list_image = get_image.list_image(path_image,path_logging);

# Exit if no image was found
if len(list_image) == 0:
#    logger.fatal('No images were found in input folder')
    sys.exit('No images were found in input folder')
    
# Exit if input path was not found
if not os.path.exists(os.path.dirname(path_image)):
#    logger.fatal('Input directory given was not found')
    sys.exit('Input directory given was not found')
    
          
def extract_feature(image):
    try:
#        for image in list_image:
        im = cv2.imread(image);
        _gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        _start_time = time.time();
        
        if method == 1: # Sift 
            method_ = 'sift/'
            sift_ = cv2.xfeatures2d_SIFT.create(nfeatures = num_feature); # Predefining number of features
            kp, des = sift_.detectAndCompute(_gray,None);
            
        if method ==2: # Surf
            method_ = 'surf/'
            surf_ = cv2.xfeatures2d.SURF_create();
            kp, des = surf_.detectAndCompute(_gray,None);
            
        if method ==3: # ORB
            method_ = 'orb/'
            orb_ = cv2.ORB_create()#nfeatures = num_feature) # Predefining number of features
            kp,des = orb_.detectAndCompute(_gray,None);    
            
        if method == 4: # Brisk
            method_ = 'brisk/'
            brisk_ = cv2.BRISK_create();
            kp, des = brisk_.detectAndCompute(_gray,None);
            
        if method == 5: # AKAZE
            method_ = 'akaze/'
            akaze_ = cv2.AKAZE_create();
            kp, des = akaze_.detectAndCompute(_gray,None);
        
        if method ==6: # Star + BRIEF
            method_ = 'StarBrief/'
            star_ = cv2.xfeatures2d.StarDetector_create()
            brief_ = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp = star_.detect(_gray,None);
            kp, des = brief_.compute(_gray, kp)
            
#           Store and Retrieve keypoint features
        temp = pickle_keypoints(kp, des)
        output = os.path.join(output_path,method_);
        
#           checking output directory if this exist otherwise will be created
        checkdir(output);
        output = output + os.path.splitext(os.path.basename(image))[0]
        
#           Saving features as numpy compressed array
        np.save(output,temp)
        
#            Printing and logging info
        _end_time = time.time()
        _time_taken = round(_end_time - _start_time,1)
        
#        logger.info('Image %s processed, took : %s sec per thread'
#        %(os.path.basename(image),str(_time_taken)))
                
#        Saving number of feature dectected to disk 
        feature = {"Number of Features" : len(kp),
                "Image" : os.path.splitext(os.path.basename(image))[0],
                "Time" : _time_taken,
                "Method" : method_}
                
        tojson(feature,path_logging + os.path.splitext(os.path.basename(image))[0] + '.json')
        
        print('finished processing %s and took %s sec per thread'
        %(os.path.splitext(os.path.basename(image))[0],_time_taken) )
        
#        Counting Features
        count_feature.append(len(kp));
        append_image.append(os.path.splitext(os.path.basename(image))[0]); 
        append_time.append(_time_taken)
        append_method.append(method_)
        
#        Reporting
        features = {"Number of Features": count_feature,
                    "Image": append_image,
                    "Time": append_time,
                    "Method": append_method};
         
    except KeyboardInterrupt:
#        logger.fatal('KeyboardInterruption, Terminated')
        sys.exit('KeyboardInterruption, Terminated')
        pause()
  
#       Returns list of images
    return features
      
def pool_handler():
    p = Pool(processes=threads,maxtasksperchild=1);
    features = list(p.imap_unordered(extract_feature, list_image))
    tojson(features,path_logging + 'extract_feature.json')
    p.close()

    
def main():
    try:
        pool_handler()
#        extract_feature(list_image)
    except KeyboardInterrupt:
#        logger.fatal('KeyboardInterruption, Terminated')
        sys.exit('KeyboardInterruption, Terminated')
        pause()
        
if __name__ == '__main__':
    try:
        _start_time = time.time();
        main()
        _overall_time = time.time();
        _time_taken_ = str(round(_overall_time - _start_time,1))
        print('Total time taken %s secs'%(_time_taken_))
#        logger.info('Total time taken is %s sec'%(_time_taken_))
        
    except KeyboardInterrupt:
#        logger.fatal('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')
        pause()
        
#Retrieve Keypoint Features
#keypoints_database = pickle.load( open( "keypoints_database.mansift", "rb" ) )
#kp1, desc1 = unpickle_keypoints(keypoints_database)


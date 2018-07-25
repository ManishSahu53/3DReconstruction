""" Matching features from images, feature_matching.py"""

import logging
reload(logging)
  
def checkdir(path):
    if path[-1] != '/':
        path = path  + '/'        
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")
    
def tojson(dictA,file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f,indent=4, separators=(',', ': '), ensure_ascii=False,encoding='utf-8')
        

def num2para(number):
    if number == 1:
        parameter_ = 'ANN/Hamming'
    return parameter_

def num2method(number):
    if number == 1:
        method_ = 'sift'
    if number == 2:
        method_ = 'surf'
    if number == 3:
        method_ = 'orb'
    if number == 4:
        method_ = 'brisk'
    if number == 5:
        method_ = 'akaze'
    if number == 6:
        method_ = 'StarBrief'
    return method_

def numfeature(path):
    list_feature = []
#    Searching of features
    for root,dirs,files in os.walk(path):
        if len(files) == 0:
            sys.exit('No features found in "%s" directory'%(path))
            break;
        for file_name in files:
            if file_name.endswith(('.npy','.NPY','Npy')):
                list_feature.append((os.path.join(path,file_name)));
    return len(list_feature)

def _convert_match_to_vector(match):
    """Convert Dmatch object to matrix form."""
    match_vector = np.zeros((len(match), 2), dtype=np.int)
    k = 0
    for mm in match:
        # mm.query will give index of mm match in keypoint detector.
        # Example- kp1[mm.queryIdx].pt will give,
        # x and y coordinate of image corresponding keypoint1
    
        match_vector[k, 0] = mm.queryIdx  # Train is master image                                         
        match_vector[k, 1] = mm.trainIdx  # Query is pair image
        k = k+1
    return match,match_vector

def robust_match_ratio_test(kp1,kp2,match,ratio):
    valid_match = []
    pts1 = []
    pts2 = []
    """Rejects matches based on popular lowe ratio test"""
    for _k,(m,n) in enumerate(match):
        if m.distance < ratio*n.distance:
            valid_match.append(m)
            pts1.append(kp1[m.queryIdx].pt) # Query is master image, Not sure
            pts2.append(kp2[m.trainIdx].pt) # Train is pair  image, Not sure
    return valid_match,np.asarray(pts1,dtype = np.int32),np.asarray(pts2,dtype = np.int32)
    
def robust_match_fundamental(pts1, pts2, valid_match):
    """Filter matches by estimating the Fundamental matrix via RANSAC."""
    if len(valid_match) < 8:
        return np.array([])
    FM_RANSAC = cv2.FM_RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, FM_RANSAC)
    index = mask.ravel() ==1
    if F[2, 2] == 0.0:
        return []
    return list(compress(valid_match, index)),pts1[index],pts2[index]   

def _convert_matches_to_vector(match):
    """Convert Dmatch object to matrix form."""
    match_vector = np.zeros((len(match), 2), dtype=np.int)
    k = 0
    for mm in match:
        match_vector[k, 0] = mm.queryIdx
        match_vector[k, 1] = mm.trainIdx
        k = k+1
    return match_vector

    
import cv2
import cPickle as pickle
import multiprocessing 
import sys,os
import argparse
import json, yaml
import numpy as np
import time 
from itertools import compress
from src.context import parallel_map
from src.dataset import unpickle_keypoints


parser = argparse.ArgumentParser(description='See description below to see all available options')

parser.add_argument('-i','--input',
                    help='Output directory feature extraction step.',
                    required=True)
                    
parser.add_argument('-o','--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
                    default = './output/',
                    required=False)
                    
parser.add_argument('-m','--method',type=int,
                    help='Methods that will be used to match '+ 
                    'features. Enter an integer value corresponding to options'+
                    ' given below. Available methods are ' + 
                    ' 1 -- ANN ' +
                    ' 2 -- BruteForce' +
                    ' [Default] is Approximate Nearest Neighbour (ANN)',
                    default = 1,
                    required=False)
                                        
parser.add_argument('-p','--parameter',type=int,
                    help='Paramter used to calculate neighbour. Available methods are'+
                    ' 1 -- Euclidian Distance/Hamming Distance' +
                    ' 2 -- Other' +
                    ' [Default] Euclidian Distance is used for SIFT,SURF. Hamming is used for Binary decriptors',
                    default = 1,
                    required=False)
                    
parser.add_argument('-f','--feature',type=int,
                    help='feature method that is used for matching'+
                    ' 1 -- SIFT ' +
                    ' 2 -- SURF'  +
                    ' 3 -- ORB'   +
                    ' 4 -- BRISK' +
                    ' 5 -- AKAZE' +
                    ' 6 -- STAR+ BRIEF'+
                    ' [Default] is SIFT',
                    default = 1,
                    required=False)
                    
parser.add_argument('-r','--ratio',type=float,
                    help='Define a ratio threshold for ratio test',
                    default = 0.8,
                    required=False)

parser.add_argument('-t','--thread',type=int,
                    help='Number of processing threads to be used ' +
                    'for multiprocessing.[Default] Using all the threads available',
                    default = 0,
                    required=False)


# Global Variable
global path_output
global path_logging
global path_report
global path_data

                  
args = parser.parse_args()

path_input = args.input;
path_output = args.output;
parameter = args.parameter; # Hamming and Euclidian distance
method = args.method; # ANN and Hamming
ratio = args.ratio;
thread = args.thread;
method_feature = num2method(args.feature); # SIFT,SURF,ORB etc

#path_input = './output/'
#path_output = './output/'
#parameter = 1
#method = 1
#ratio = 0.8
#method_feature = num2method(1)
#thread = multiprocessing.cpu_count();

# Saving
saving_matches = 'matching_feature'

# Update path_output and output directories
path_output = os.path.join(path_output,saving_matches)       
path_logging = os.path.join(path_output,'logging') # Individual file record
path_report = os.path.join(path_output,'report') # Summary of whole file
path_data = os.path.join(path_output,'data') # Any data to be saved

# Checking if path  exists, otherwise will be created
checkdir(path_output)
checkdir(path_logging)
checkdir(path_report)
checkdir(path_data)

# Finding exif folder containing exif, imagepair, descriptors
path_exif = os.path.join(path_input,'exif','data')
path_feature =os.path.join(path_input,'extract_feature','data',method_feature)

# Json files
file_exif = os.path.join(path_exif,'exif.json')
file_imagepair = os.path.join(path_exif,'imagepair.json')


# loading exif and image pairs details
exif = yaml.safe_load(open(file_exif))
imagepair = yaml.safe_load(open(file_imagepair))


# Number of feature and number of images should be same
_num_feature = numfeature(path_feature)
_num_image = len(imagepair)
_num_pair = len(imagepair[0])

# If number of images is not equal to number of features extract.
if _num_feature != _num_image:
    sys.exit('Number of features and number of images are not same. Check if features extract ran properly')
    
# Cheching if sift_extract_feature.json and imagepair.json exists or not
if os.path.isfile(file_exif):
    print('exif.json found')
else:
    sys.exit('No exif.json found. extract_feature.py command first before running matching_feature.py')

# Checking if imagepair.json exist or not
if os.path.isfile(file_imagepair):
    print('imagepair.json found')
else:
    sys.exit('No imagepair.json found. Run exif.py command first before running matching_feature.py')
    
# Allowing only euclidian/hamming distance to be used
if parameter != 1:
    sys.exit('Other matching matrices not implemented. Use Euclidian/hamming distance')    

# Number of threads to maximum if not assigned any.
if thread ==0:
    thread = multiprocessing.cpu_count();
    
elif thread>multiprocessing.cpu_count():
    thread = multiprocessing.cpu_count();
    print('Number of threads assigned (%s) is greater than maximum available (%s)'%(thread,multiprocessing.cpu_count()))
    
#if args.feature == 1 or 2:
#    similarity_matric = 'euclidian' # Non binary
#else:
#    similarity_matric = 'hamming' # For binary
  
image = []
valid_match_num = []
invalid_match_num = []
inlier_ratio = []
match_num = []
match_report = []

def match_feature(imagepair):#,path_feature,path_output):

    try:
        match_data = []
#        for i in range(_num_image):
        _pair_logging = [];
        _start_time = time.time()
        master_im = imagepair[0];
        _match = {};
        for j in range(1,_num_pair):
            pair_im = imagepair[j];
            valid_match = []
            
#                Retrieve Keypoint Features
            kp1, des1, color = unpickle_keypoints(master_im,path_feature)
            kp2, des2, color = unpickle_keypoints(pair_im,path_feature)
            
#            Changing matching method for SIFT/SURF and Binary features.
#            Binary doesn't have ANN option and so Brute Force method is used.
            
            if args.feature == 1 or args.feature == 2:
#                FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50)   # or pass empty dictionary
            
#                loading saved keypoints and descriptors
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                match_ = flann.knnMatch(des1,des2,k=2)
                
#                store all the good matches as per Lowe's ratio test.
                valid_match,pts1,pts2 = robust_match_ratio_test(kp1,kp2,match_,ratio)
                
#                Robut fundamental outlier removal
                valid_match,pts1,pts2 = robust_match_fundamental(pts1,pts2,valid_match)
                match_index = np.array(_convert_matches_to_vector(valid_match),dtype=int)

                
            elif args.feature == 3 or args.feature == 4 or args.feature == 5 or args.feature == 6: # For binary descriptors, using hamming distance
                
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

                # Match descriptors.
                match_ = bf.knnMatch(des1,des2,k=2)
                
#                store all the good matches as per Lowe's ratio test.
                valid_match,pts1,pts2 = robust_match_ratio_test(kp1,kp2,match_,ratio)
                
#                Robut fundamental outlier removal
                valid_match,pts1,pts2 = robust_match_fundamental(pts1,pts2,valid_match)
                match_index = np.array(_convert_matches_to_vector(valid_match),dtype = int)

#                Printing and logging info
#            logger.info('Image %s processed, took : %s sec per thread'
#            %(os.path.basename(image),str(_time_taken)))
            
            _pair_logging.append({"Pair im" : pair_im,
                                 "Total matches" : len(match_),
                                 "Valid matches" : len(valid_match),
                                 "Inlier percentage" : round(float(len(valid_match))*100,2)/len(match_)})
            
#           Appending matches data for saving 
            match_data.append([[valid_match,match_index],[pts1,pts2],[master_im,pair_im]])
            _match[pair_im] = match_index
            
#         Updating path_data to path_feature
        print(master_im)
        path_match = os.path.join(path_data,method_feature);
        checkdir(path_match)
        
#         Saving features as numpy compressed array
        path_match = os.path.join(path_match, os.path.splitext(os.path.basename(master_im))[0])
#        np.save(path_match,match_data)_match
        np.save(path_match,_match)
        
        _end_time = time.time()
        _time_taken = round(_end_time - _start_time,1) 
        

        _master_logging = {"Master Image" : master_im,
                "Pair Images" : _pair_logging,
                "Features extraction method" : method_feature,
                "Threshold Ration" : ratio,
                "Similarity matric" : num2para(parameter),
                "Time" : _time_taken};
                
#         Saving corresponding matches of master image to file 
        tojson(_master_logging,os.path.join(path_logging,os.path.splitext(os.path.basename(master_im))[0] + '.json'))
   
        match_report.append(_master_logging)
           
    except KeyboardInterrupt:
#        logger.fatal('KeyboardInterruption, Terminated')
        sys.exit('KeyboardInterruption, Terminated')
        pause()
    return match_report
    
def pool_match_feature():
    try:
        match = list(parallel_map(match_feature,imagepair,thread))#,path_feature,path_output))
        tojson(match,os.path.join(path_report, method_feature+ '_match_feature.json'))
        
    except KeyboardInterrupt:
        sys.exit('KeyboardInterruption, Terminated')
        pause()

def main():
    try:
        pool_match_feature()
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
#kp1, desc1, color = unpickle_keypoints(keypoints_database)

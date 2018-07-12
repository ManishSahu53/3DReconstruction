""" Matching features from images, feature_matching.py"""

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id)#, descriptors[i])     
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(image,path_feature):

    image = os.path.splitext(os.path.basename(image))[0];
    image = os.path.join(path_feature,image+'.npy');
    array = pickle.load( open( image, "rb" ) );
    
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

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
    
import cv2
import cPickle as pickle
import multiprocessing 
from multiprocessing import Pool
import sys,os
import argparse
import json, yaml
import numpy as np

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
                    ' 1 -- Eucledian Distance' +
                    ' 2 -- Hamming Distance' +
                    ' [Default] is Eucledian Distance',
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
args = parser.parse_args()

path_input = args.input;
path_output = args.output;
parameter = args.parameter;
method = args.method;
ratio = args.ratio;

method_feature = num2method(args.feature);

# output folder containing exif and imagepair
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

if _num_feature != _num_image:
    sys.exit('Number of features and number of images are not same. Check if features extract ran properly')
    
# Cheching if sift_extract_feature.json and imagepair.json exists or not
if os.path.isfile(file_exif):
    print('exif.json found')
else:
    sys.exit('No exif.json found. extract_feature.py command first before running matching_feature.py')

if os.path.isfile(file_imagepair):
    print('imagepair.json found')
else:
    sys.exit('No imagepair.json found. Run exif.py command first before running matching_feature.py')



def match_feature(imagepair,path_feature,path_output):
    for i in range(_num_image):
        for j in range(1,_num_pair):
            master_im = imagepair[i][0];
            pair_im = imagepair[i][j];
            
#            Retrieve Keypoint Features
            kp1, desc1 = unpickle_keypoints(master_im,path_feature)
            
        #      FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
        
        #     loading saved keypoints and descriptors
            kp1, desc1 = unpickle_keypoints(keypoints_database)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append(m)
            
            
def pool_match_feature():
    try:
        p = Pool(processes=threads,maxtasksperchild=1);
        features = list(p.imap_unordered(extract_feature, list_image))
        tojson(features,os.path.join(path_report, num2method(method), '_match_feature.json'))
        p.close()
        
    except KeyboardInterrupt:
        sys.exit('KeyboardInterruption, Terminated')
        pause()

def main():
    try:
        pool_extract_feature()
#        pool_match_feature()
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

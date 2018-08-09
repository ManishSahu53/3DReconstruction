""" Extracting features from images, extract_feature.py"""
import logging
reload(logging)


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
        method_ = 'starbrief'
    return method_


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


import cv2
import numpy as np
import os
import time
import sys
import yaml
import argparse
from src import get_image
from src.context import parallel_map
from src.dataset import pickle_keypoints
from src.dataset import checkdir, tojson, kp2xy
from src import dataset
from src import feature as f
import traceback

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-i', '--input',
                    help='Input directory containing images.',
                    required=True)

parser.add_argument('-o', '--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
                    default='./output/',
                    required=False)

args = parser.parse_args()

# parsing inputs
path_image = args.input
path_output = args.output

# loading parameter file
file_para = dataset.para_lo(path_output)
para = yaml.safe_load(open(file_para))

# defining extractor and minimum features
method = para['feature_extractor']
num_feature = para['feature_min']
print('method using : ' + method, 'num of features : ' + str(num_feature))

path_feature = dataset.feature_lo(path_output,method)
# output directories
path_output = path_feature[3]
path_logging = path_feature[2]
path_report = path_feature[1]
path_data = path_feature[0]

number = f.method2num(method)

# Check if method > 5
if number > 6 or number < 1:
    sys.exit('Invalid method selected. Only %s methods are available. See help to know how to use them ' % (str(6)))


# Converting to realative path
path_image = os.path.realpath(path_image)

# Initialising Parameters
count_feature = []
append_image = []
append_time = []
append_method = []
thread = 6

# Getting list of images present in given folder.
list_image = get_image.list_image(path_image, path_logging)

# Exit if no image was found
if len(list_image) == 0:
    sys.exit('No images were found in input folder')

# Exit if input path was not found
if not os.path.exists(os.path.dirname(path_image)):
    sys.exit('Input directory given was not found')


def extract_feature(image):
    try:
        im = cv2.imread(image)
        _gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _start_time = time.time()

        if method == 'sift':  # Sift
            if num_feature == 0:
                sift_ = cv2.xfeatures2d_SIFT.create()
            else:
                sift_ = cv2.xfeatures2d_SIFT.create(
                    nfeatures=num_feature)  # Predefining number of features

            kp, des = sift_.detectAndCompute(_gray, None)
            pt = kp2xy(kp)
            x = pt[:,0].round().astype(int)
            y = pt[:,1].round().astype(int)
            color = im[y, x]  # Since im[row,column]

        if method == 'surf':  # Surf
            if num_feature == 0:
                surf_ = cv2.xfeatures2d.SURF_create()
            else:
                surf_ = cv2.xfeatures2d.SURF_create(2000)  # ~ 50000 features

            kp, des = surf_.detectAndCompute(_gray, None)
            pt = kp2xy(kp)
            
            x = pt[:,0].round().astype(int)
            y = pt[:,1].round().astype(int)
            color = im[y, x]  # Since im[row,column]

        if method == 'orb':  # ORB
            if num_feature == 0:
                orb_ = cv2.ORB_create()
            else:
                # Predefining number of features
                orb_ = cv2.ORB_create(nfeatures=num_feature)

            kp, des = orb_.detectAndCompute(_gray, None)
            
            pt = kp2xy(kp)
            x = pt[:,0].round().astype(int)
            y = pt[:,1].round().astype(int)
            color = im[y, x]  # Since im[row,column]

        if method == 'brisk':  # Brisk
            if num_feature == 0:
                brisk_ = cv2.BRISK_create()
            else:
                brisk_ = cv2.BRISK_create(75)  # ~ 50,000 features

            kp, des = brisk_.detectAndCompute(_gray, None)
            
            pt = kp2xy(kp)
            x = pt[:,0].round().astype(int)
            y = pt[:,1].round().astype(int)
            color = im[y, x]  # Since im[row,column]

        if method == 'akaze':  # AKAZE
            akaze_ = cv2.AKAZE_create()
            kp, des = akaze_.detectAndCompute(_gray, None)

            pt = kp2xy(kp)
            x = pt[:,0].round().astype(int)
            y = pt[:,1].round().astype(int)
            color = im[y, x]  # Since im[row,column]

        if method == 'starbrief':  # Star + BRIEF
            star_ = cv2.xfeatures2d.StarDetector_create()  # ~ 80,000 features
            brief_ = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp = star_.detect(_gray, None)
            kp, des = brief_.compute(_gray, kp)

            pt = kp2xy(kp)
            x = pt[:,0].round().astype(int)
            y = pt[:,1].round().astype(int)
            color = im[y, x]  # Since im[row,column]

           #Store and Retrieve keypoint features
        temp = pickle_keypoints(kp, des, color)
        
        path_feature = path_data
           #Saving features as numpy compressed array
        path_feature = os.path.join(path_feature,
            os.path.splitext(os.path.basename(image))[0])
        np.save(path_feature, temp)

            #Printing and logging info
        _end_time = time.time()
        _time_taken = round(_end_time - _start_time, 1)


        #Saving number of feature dectected to disk
        feature = {"Number of Features": len(kp),
                   "Image": os.path.splitext(os.path.basename(image))[0],
                   "Time": _time_taken,
                   "Method": method}

        tojson(feature, os.path.join(path_logging, os.path.splitext(
            os.path.basename(image))[0] + '.json'))

        print('finished processing %s and took %s sec per thread'
              % (os.path.splitext(os.path.basename(image))[0], _time_taken))

#        Counting Features
        count_feature.append(len(kp))
        append_image.append(os.path.splitext(os.path.basename(image))[0])
        append_time.append(_time_taken)
        append_method.append(method)

#        Reporting
        features = {"Number of Features": count_feature,
                    "Image": append_image,
                    "Time": append_time,
                    "Method": append_method}

    except:
        print 'Exception in '+extract_feature.__name__
        traceback.print_exc()

#       Returns list of images
    return features


def pool_extract_feature():
    try:
        features = list(parallel_map(extract_feature, list_image, thread))
        tojson(features, os.path.join(path_report, method + '_extract_feature.json'))

    except KeyboardInterrupt:
        sys.exit('KeyboardInterruption, Terminated')
        pause()


def main():
    try:
        pool_extract_feature()

    except KeyboardInterrupt:
        sys.exit('KeyboardInterruption, Terminated')
        pause()


if __name__ == '__main__':
    try:
        _start_time = time.time()
        main()
        _overall_time = time.time()
        _time_taken_ = str(round(_overall_time - _start_time, 1))
        print('Total time taken %s secs' % (_time_taken_))

    except KeyboardInterrupt:
        sys.exit('Interrupted by keyboard')
        pause()

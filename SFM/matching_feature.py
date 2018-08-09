""" Matching features from images, feature_matching.py"""

import logging
reload(logging)

import cv2
import cPickle as pickle
import multiprocessing
import sys
import os
import argparse
import yaml
import numpy as np
import time
from src.context import parallel_map
from src.dataset import unpickle_keypoints, checkdir, tojson, pause, para_lo, feature_lo, exif_lo
from src.matching import dict2list, method2num, numfeature, num2para
from src.matching import robust_match_ratio_test, _convert_match_to_vector, robust_match_fundamental
import traceback

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-i', '--input',
                    help='Input directory of images.',
                    required=True)

parser.add_argument('-o', '--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
                    default='./output/',
                    required=False)

parser.add_argument('-t', '--thread', type=int,
                    help='Number of processing threads to be used ' +
                    'for multiprocessing.[Default] Using all the threads available',
                    default=0,
                    required=False)


# Global Variable
global path_output
global path_logging
global path_report
global path_data


args = parser.parse_args()

path_input = args.input
path_output = args.output
thread = args.thread


#path_input = './output/'
#path_output = './output/'
#para = 1
#method = 1
#ratio = 0.8
#method_feature = num2method(1)
#thread = multiprocessing.cpu_count();

# Saving
saving_matches = 'matching_feature'

# Update path_output and output directories
path_output = os.path.join(path_output, saving_matches)
path_logging = os.path.join(path_output, 'logging')  # Individual file record
path_report = os.path.join(path_output, 'report')  # Summary of whole file
path_data = os.path.join(path_output, 'data')  # Any data to be saved

# Checking if path  exists, otherwise will be created
checkdir(path_output)
checkdir(path_logging)
checkdir(path_report)
checkdir(path_data)


# reading parameters values 
file_para = para_lo(args.output)

para = yaml.safe_load(open(file_para))
ratio = para['lowe_ratio']
method_feature = para['feature_extractor']
num_feature = method2num(method_feature) 
method = para['matcher_type']

# Finding exif folder containing exif, imagepair, descriptors
path_exif = exif_lo(args.output)
path_feature = feature_lo(args.output,method_feature)

# Json files
file_exif = path_exif[0]
file_imagepair = path_exif[1]


# loading exif and image pairs details
exif = yaml.safe_load(open(file_exif))
imagepair = yaml.safe_load(open(file_imagepair))
imagepair = dict2list(imagepair)


# Number of feature and number of images should be same
_num_feature = numfeature(path_feature[0])
_num_image = len(exif.keys())
_num_pair = len(imagepair[0])

# If number of images is not equal to number of features extract.
if _num_feature != _num_image:
    print('Number of features: %s and number of images: %s'%(_num_feature,_num_image))
    sys.exit('Number of features and number of images are not same. Check if features extract ran properly')

# Cheching if sift_extract_feature.json and imagepair.json exists or not
if os.path.isfile(file_exif):
    print('exif.json found')
else:
    sys.exit(
        'No exif.json found. extract_feature.py command first before running matching_feature.py')

# Checking if imagepair.json exist or not
if os.path.isfile(file_imagepair):
    print('imagepair.json found')
else:
    sys.exit(
        'No imagepair.json found. Run exif.py command first before running matching_feature.py')


# Number of threads to maximum if not assigned any.
if thread == 0:
    thread = multiprocessing.cpu_count()

elif thread > multiprocessing.cpu_count():
    thread = multiprocessing.cpu_count()
    print('Number of threads assigned (%s) is greater than maximum available (%s)' % (
        thread, multiprocessing.cpu_count()))

image = []
valid_match_num = []
invalid_match_num = []
inlier_ratio = []
match_num = []
match_report = []


def match_feature(imagepair):  # ,path_feature,path_output):
    try:
        match_data = []
        _pair_logging = []
        _start_time = time.time()
        master_im = imagepair[0]
        print('Processing %s image' % (master_im))
        _match = {}
        for j in range(0, _num_pair):
            pair_im = imagepair[j]

            # Since first row of image corresponds to the imgae itself,
            # So if pair_im == master_im then skip this matching

            if pair_im == master_im:
                continue
            valid_match = []
#                Retrieve Keypoint Features
            kp1, des1, color = unpickle_keypoints(master_im, path_feature[0])
            kp2, des2, color = unpickle_keypoints(pair_im, path_feature[0])

#            Changing matching method for SIFT/SURF and Binary features.
#            Binary doesn't have ANN option and so Brute Force method is used.

            if num_feature == 1 or num_feature == 2:
                #                FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)   # or pass empty dictionary

#                loading saved keypoints and descriptors
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                match_ = flann.knnMatch(des1, des2, k=2)

#                store all the good matches as per Lowe's ratio test.
                valid_match, pts1, pts2 = robust_match_ratio_test(
                    kp1, kp2, match_, ratio)

#                Robut fundamental outlier removal
                valid_match, pts1, pts2 = robust_match_fundamental(
                    pts1, pts2, valid_match)
                match_index = np.array(
                    _convert_match_to_vector(valid_match), dtype=int)

            # For binary descriptors, using hamming distance
            elif num_feature == 3 or num_feature == 4 or num_feature == 5 or num_feature == 6:

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

                # Match descriptors.
                match_ = bf.knnMatch(des1, des2, k=2)

#                store all the good matches as per Lowe's ratio test.
                valid_match, pts1, pts2 = robust_match_ratio_test(
                    kp1, kp2, match_, ratio)

#                Robut fundamental outlier removal
                valid_match, pts1, pts2 = robust_match_fundamental(
                    pts1, pts2, valid_match)
                match_index = np.array(
                    _convert_match_to_vector(valid_match), dtype=int)

            _pair_logging.append({"Pair im": pair_im,
                                  "Total matches": len(match_),
                                  "Valid matches": len(valid_match),
                                  "Inlier percentage": round(float(len(valid_match))*100, 2)/len(match_)})

#           Appending matches data for saving
            match_data.append([[valid_match, match_index], [
                              pts1, pts2], [master_im, pair_im]])
            _match[pair_im] = match_index

#         Updating path_data to path_feature
        path_match = os.path.join(path_data, method_feature)
        checkdir(path_match)

#         Saving features as numpy compressed array
        path_match = os.path.join(
            path_match, os.path.splitext(os.path.basename(master_im))[0])
        np.save(path_match, _match)

        _end_time = time.time()
        _time_taken = round(_end_time - _start_time, 1)

        _master_logging = {"Master Image": master_im,
                           "Pair Images": _pair_logging,
                           "Features extraction method": method_feature,
                           "Threshold Ration": ratio,
                           "Time": _time_taken}

#         Saving corresponding matches of master image to file
        tojson(_master_logging, os.path.join(path_logging,
                                             os.path.splitext(os.path.basename(master_im))[0] + '.json'))

        match_report.append(_master_logging)

    except Exception or KeyboardInterrupt:
        #        logger.fatal('KeyboardInterruption, Terminated')
        traceback.print_exc()
        sys.exit('Interruption in parallel processing, Terminated')
    return match_report


def pool_match_feature():
    try:
        match = list(parallel_map(match_feature, imagepair, thread))
        tojson(match, os.path.join(path_report,
                                   method_feature + '_match_feature.json'))

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
        _start_time = time.time()
        main()
        _overall_time = time.time()
        _time_taken_ = str(round(_overall_time - _start_time, 1))
        print('Total time taken %s secs' % (_time_taken_))
#        logger.info('Total time taken is %s sec'%(_time_taken_))

    except KeyboardInterrupt:
        #        logger.fatal('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')
        pause()

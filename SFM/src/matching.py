""" Matching features from images, feature_matching.py"""

import logging
reload(logging)
import numpy as np
from itertools import compress
import json,yaml,cv2,os,sys

def dict2list(dictio):
    temp = []
    for key in dictio.keys():
        neighbour = dictio[key]
        neighbour.insert(0,key)
        temp.append(neighbour)
    return temp    
        
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
    return match_vector

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
    
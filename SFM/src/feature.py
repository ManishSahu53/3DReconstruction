""" Extracting features from images, extract_feature.py"""
import logging
reload(logging)
import cv2
import os
import json
import sys
import numpy as np


def checkdir(path):
    if path[-1] != '/':
        path = path + '/'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))


def pickle_keypoints(keypoints, descriptors, color):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    return [temp_array, [color]]


def unpickle_keypoints(_array):
    keypoints = []
    descriptors = []
    color = _array[1]
    array = _array[0]

    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors), color


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

def method2num(method):
    if method == 'sift':
        number = 1
    if method == 'surf':
        number = 2
    if method == 'orb':
        number = 3
    if method == 'brisk':
        number = 4
    if method == 'akaze':
        number = 5
    if method == 'starbrief':
        number = 6
    return number

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


def tojson(dictA, file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f, indent=4, separators=(',', ': '),
                  ensure_ascii=False, encoding='utf-8')


def create_logger(name):
    """Create a logger

    args: name (str): name of logger

    returns: logger (obj): logging.Logger instance
    """
    fmt = logging.Formatter('%(asctime)s - %(name)s -'
                            ' %(levelname)s -%(message)s')
    hdl = logging.FileHandler(name+'.log')

    return logging


def kp2xy(kp):
    point = np.array([(i.pt[0], i.pt[1]) for i in kp])
    x = point[:, 0].round().astype(int)
    y = point[:, 1].round().astype(int)
    return x, y


def normalized_image_coordinates(pixel_coords, width, height):
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p


def denormalized_image_coordinates(norm_coords, width, height):
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p
    


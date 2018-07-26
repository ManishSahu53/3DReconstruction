""" Dataset function which contains location information"""

import os 
import numpy as np
import cv2,json
import cPickle as pickle

def kp2xy(kp):
    point = np.array([(i.pt[0], i.pt[1]) for i in kp])
    return point
    
def pickle_keypoints(keypoints, descriptors,color):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        i = i +1
        temp_array.append(temp)
    return [temp_array,[color]]  
    
def unpickle_keypoints(image,path_feature):
    path_feature = os.path.realpath(path_feature);
    
    image = os.path.splitext(os.path.basename(image))[0];
    image = os.path.join(path_feature,image+'.npy');
    array = np.load( open( image, "rb" ) );

    keypoints = []
    descriptors = []
    color = array[1][0]
    array = array[0]
    
    for point in array:
        temp_feature = cv2.KeyPoint(point[0][0],point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors),color
    
    
# Checking if directory exist
def checkdir(path):
    if path[-1] != '/':
        path = path  + '/'        
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))

# exif file location
def exif_lo(path_output):
    saving_folder = 'exif'
    path_output = os.path.join(path_output,saving_folder)
    path_logging = os.path.join(path_output,'logging')
    path_data = os.path.join(path_output,'data')
    file_exif = os.path.join(path_data,'exif.json')
    file_imagepair = os.path.join(path_data,'imagepair.json')
    # Checking if path  exists, otherwise will be created
    checkdir(path_output);
    checkdir(path_logging);
    checkdir(path_data);
    return file_exif,file_imagepair, path_data, path_logging, path_output

# features location
def feature_lo(path_output,method_feature):
    saving_feature = 'extract_feature'
    # Update path_output and output directories
    path_output = os.path.join(path_output,saving_feature)       
    path_logging = os.path.join(path_output,'logging')
    path_report = os.path.join(path_output,'report')
    path_data = os.path.join(path_output,'data',method_feature)
    
    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return path_data, path_report, path_logging, path_output

# mactchin location    
def match_lo(path_output,method_feature):
    saving_matches = 'matching_feature'

    path_output = os.path.join(path_output,saving_matches)       
    path_logging = os.path.join(path_output,'logging') # Individual file record
    path_report = os.path.join(path_output,'report') # Summary of whole file
    path_data = os.path.join(path_output,'data',method_feature) # Any data to be saved
    
    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return path_data, path_report, path_logging, path_output

# track location
def track_lo(path_output):
    saving_track = 'track'
    
    path_output = os.path.join(path_output,saving_track)       
    path_logging = os.path.join(path_output,'logging') # Individual file record
    path_report = os.path.join(path_output,'report') # Summary of whole file
    path_data = os.path.join(path_output,'data') # Any data to be saved
    file_track = os.path.join(path_data,'track.json')
    
    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return file_track, path_data, path_report, path_logging, path_output

# loading matches
def load_match(file_match):
    match = np.load(open(file_match)).item()
    return match

def tojson(dictA,file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f,indent=4, separators=(',', ': '), ensure_ascii=False,encoding='utf-8')
        
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

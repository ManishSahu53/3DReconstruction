""" Dataset function which contains location information"""

import os 
import numpy as np
import cv2,json
import cPickle as pickle
from src import types
from six import iteritems

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
    file_camera_model = os.path.join(path_data,'camera.json')
    
    # Checking if path  exists, otherwise will be created
    checkdir(path_output);
    checkdir(path_logging);
    checkdir(path_data);
    return file_exif,file_imagepair,file_camera_model, path_data, path_logging, path_output


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
def track_lo(path_output,method_feature):
    saving_track = 'track'
    
    path_output = os.path.join(path_output,saving_track)       
    path_logging = os.path.join(path_output,'logging') # Individual file record
    path_report = os.path.join(path_output,'report') # Summary of whole file
    path_data = os.path.join(path_output,'data',method_feature) # Any data to be saved
    file_track = os.path.join(path_data,'track.csv')
    
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

### Cameras ###

def camera_to_json(camera):
    """
    Write camera to a json object
    """
    if camera.projection_type == 'brown':
        return {
            'projection_type': camera.projection_type,
            'width': camera.width,
            'height': camera.height,
            'focal_x': camera.focal_x,
            'focal_y': camera.focal_y,
            'c_x': camera.c_x,
            'c_y': camera.c_y,
            'k1': camera.k1,
            'k2': camera.k2,
            'p1': camera.p1,
            'p2': camera.p2,
            'k3': camera.k3,
            'focal_x_prior': camera.focal_x_prior,
            'focal_y_prior': camera.focal_y_prior,
            'c_x_prior': camera.c_x_prior,
            'c_y_prior': camera.c_y_prior,
            'k1_prior': camera.k1_prior,
            'k2_prior': camera.k2_prior,
            'p1_prior': camera.p1_prior,
            'p2_prior': camera.p2_prior,
            'k3_prior': camera.k3_prior
        }
        raise NotImplementedError

def camera_from_json(key, obj):
    """
    Read camera from a json object
    """
#    Only brownperspective cameras

    camera = types.BrownPerspectiveCamera()
    camera.id = key
    camera.width = obj.get('width', 0)
    camera.height = obj.get('height', 0)
    camera.focal_x = obj['focal_x']
    camera.focal_y = obj['focal_y']
    camera.c_x = obj.get('c_x', 0.0)
    camera.c_y = obj.get('c_y', 0.0)
    camera.k1 = obj.get('k1', 0.0)
    camera.k2 = obj.get('k2', 0.0)
    camera.p1 = obj.get('p1', 0.0)
    camera.p2 = obj.get('p2', 0.0)
    camera.k3 = obj.get('k3', 0.0)
    camera.focal_x_prior = obj.get('focal_x_prior', camera.focal_x)
    camera.focal_y_prior = obj.get('focal_y_prior', camera.focal_y)
    camera.c_x_prior = obj.get('c_x_prior', camera.c_x)
    camera.c_y_prior = obj.get('c_y_prior', camera.c_y)
    camera.k1_prior = obj.get('k1_prior', camera.k1)
    camera.k2_prior = obj.get('k2_prior', camera.k2)
    camera.p1_prior = obj.get('p1_prior', camera.k1)
    camera.p2_prior = obj.get('p2_prior', camera.k2)
    camera.k3_prior = obj.get('k3_prior', camera.k1)
    return camera

def cameras_from_json(obj):
    """
    Read cameras from a json object
    """
    cameras = {}
    for key, value in iteritems(obj):
        cameras[key] = camera_from_json(key, value)
    return cameras
    
def load_camera_models(file_camera):
    obj = json.load(open(file_camera))
    return cameras_from_json(obj)
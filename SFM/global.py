""" Global  bundle adjustment """

def load_match(path_output, method_feature, remaining_images):
    path_match = dataset.match_lo(path_output, method_feature)
    match = {}
    for root, dirs, files in os.walk(path_match[0]):
        if len(files) == 0:
            print('No files found in "%s" directory' % (path_match[0]))
            sys.exit('No files found')
            break
        for file_name in files:
            if file_name.endswith(('.npy', '.Npy', '.NPY')):
                count = 0
                count_im = 0
                im1_match = np.load(os.path.join(
                    path_match[0], file_name)).item()
                im1 = os.path.splitext(file_name)[0] + '.JPG'
                intersect =  set.intersection(set([im1]),remaining_images)
                # We want to only process uncalibrated images
                if len(intersect) ==0:
                    continue
                for im2 in im1_match:
                    intersect =  set.intersection(set([im2]),remaining_images)
                    
                    # dont want to match uncalibrated images with uncalibrated images
                    if len(intersect) ==0:
                        count_im = count_im + 1
                        count = len(im1_match[im2]) + count
                match[im1] =  count/count_im
    return match


        
        
from src import track
from src import dataset
from src import global_ba as gba
import yaml
import os
import sys
import argparse
from src.matching import num2method
import numpy as np
import time
from six import iteritems
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


start_time = time.time()
#path_input = '../test_dataset/Images/'
#path_output = './output'
#method_feature = 'sift'

parser = argparse.ArgumentParser(description='See description below to see all available options')

parser.add_argument('-i','--input',
                    help='Input directory containing images.',
                    required=True)

parser.add_argument('-o','--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
                    default = './output/',
                    required=False)

args = parser.parse_args()

path_input = args.input;
path_output = args.output;

# getting files
file = dataset.exif_lo(path_output)
file_ref = dataset.ref_lo(path_output)
file_para = dataset.para_lo(path_output)
para = yaml.safe_load(open(file_para))

method_feature = para['feature_extractor']

path_reconstruction = dataset.reconstruction_lo(path_output, method_feature)

# Cheching if exif.json and imagepair.json exists or not

if os.path.isfile(file[0]):
    print('exif.json found')
else:
    sys.exit('No exif.json found. extract_feature.py command first before running')

# Checking if imagepair.json exist or not
if os.path.isfile(file[1]):
    print('imagepair.json found')
else:
    sys.exit('No imagepair.json found. Run exif.py command first before running')

# Checking if imagepair.json exist or not
if os.path.isfile(file_ref):
    print('reference.json found')
else:
    sys.exit('No reference.json found. Run exif.py command first before running')

if os.path.isfile(file_para):
    print('parameter.json found')
else:
    sys.exit('No parameter.json found. Run exif.py command first before running')

# loading files
exif = yaml.safe_load(open(file[0]))
imagepair = yaml.safe_load(open(file[1]))
ref = yaml.safe_load(open(file_ref))
gcp = None

_im = exif.keys()
width  = exif[_im[0]]['width'] 
height = exif[_im[0]]['height']

# Creating camera class
camera_model = dataset.load_camera_models(file[2])
no_im = len(exif)

# Getting file track
file_track = dataset.track_lo(path_output, method_feature)

# Loading Track Graph
track_graph = track.load_track_graph(file_track[0])

# Extracting images and tracks
tracks, images = track.tracks_and_images(track_graph)
remaining_images = set(images)

# Taking common features
common_track = track.all_common_tracks(track_graph, tracks)

#  Get good image pair
processes = 16
pair = gba.compute_image_pair(common_track, [exif, camera_model], para)
num_pair = len(pair)

rec_report = {}
reconstructions = []

for im1, im2 in pair:
    # Since first row of image pair is always that image itself,
    # so there is cases where im1==im2 which should be avoided.
    if im1 == im2:
        continue

    if im1 in remaining_images and im2 in remaining_images:
        # Selectin common tracks between images
        tracks, p1, p2 = common_track[im1, im2]
        print('selecting pairs...')
        reconstruction, rec_report['bootstrap'] = gba.bootstrap_reconstruction(
            camera_model, exif, ref, track_graph, im1, im2, p1, p2, para)
        print('pairs selected...')
        if reconstruction:
            remaining_images.remove(im1)
            remaining_images.remove(im2)
            print('Growing reconstruction ...')
            reconstruction, rec_report['grow'] = gba.grow_reconstruction(
                para, track_graph, reconstruction, remaining_images, gcp, exif, ref)
            print('Grow completed')
            reconstructions.append(reconstruction)
            reconstructions = sorted(reconstructions,
                                     key=lambda x: -len(x.shots))
                                         
reconstruct = dataset.save_reconstruction(reconstructions, path_reconstruction[0])
dataset.save_ply(reconstructions, path_reconstruction[0])

"""Simple BA"""
point_3d, pt_color, point_2d, camera_index, point_index = gba.arr_param(reconstruct, track_graph, remaining_images)

# R, T, f, cx, cy, k1, k2, k3, p1, p2
camera_params = gba.cam_param(camera_index, reconstruct)


n_cameras = camera_params.shape[0]
n_points = point_3d.shape[0]

n = 14 * n_cameras + 3 * n_points
m = 2 * point_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))


#%matplotlib inline
x0 = np.hstack((camera_params.ravel(), point_3d.ravel()))

f0 = gba.fun(x0, n_cameras, n_points, camera_index, point_index, point_2d, camera_model)

plt.plot(f0)

A = gba.bundle_adjustment_sparsity(n_cameras, n_points, camera_index, point_index)

t0 = time.time()
res = least_squares(gba.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss = 'soft_l1',
                    args=( n_cameras, n_points, camera_index, point_index, point_2d, camera_model))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))
plt.plot(res.fun)


# Extracting matches of uncalibrated images
average_match = load_match(path_output,method_feature,remaining_images)
end_time = time.time()

reconstruct_report = {}
uncalibrated_report = {}
reproj_error = []
num_uncalibrated = len(remaining_images)
total_image = len(images)
num_calibrated = total_image- num_uncalibrated

num_segment = len(reconstruct)
for i in range(num_segment):
    for j in reconstruct[i]['points'].keys():
        reproj_error.append(reconstruct[i]['points'][j]['reprojection_error'])
reproj_error = np.asarray(reproj_error)

reconstruct_report['total_image'] = total_image
reconstruct_report['num_uncalibrated'] = num_uncalibrated
reconstruct_report['avg_reprojection_error'] = round(np.mean(reproj_error),3)
reconstruct_report['max_reprojection_error'] = round(np.max(reproj_error),3)
reconstruct_report['total_time (secs)'] = round(end_time-start_time,0)

uncalibrated_report['average_match'] =  average_match

dataset.tojson(reconstruct_report,os.path.join(path_reconstruction[1],'reconstruction_report.json'))
dataset.tojson(rec_report,os.path.join(path_reconstruction[1],'technical_report.json'))
dataset.tojson(uncalibrated_report,os.path.join(path_reconstruction[1],'uncalibrated_images.json'))
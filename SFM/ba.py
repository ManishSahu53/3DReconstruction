""" Incremental bundle adjustment """
from src import track
from src import dataset
from src import incremental_ba as iba
import yaml
import os
import sys

path_input = './input'
path_output = './output'
method_feature = 'sift'

file = dataset.exif_lo(path_output)
file_ref = dataset.ref_lo(path_output)
file_para = dataset.para_lo(path_output)

# path_data, path_report, path_logging, path_output
path = dataset.reconstruction_lo(path_output, method_feature)

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

exif = yaml.safe_load(open(file[0]))
imagepair = yaml.safe_load(open(file[1]))
ref = yaml.safe_load(open(file_ref))
parameter = yaml.safe_load(open(file_para))
gcp = None

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
pair = iba.compute_image_pair(common_track, [exif, camera_model], parameter)
num_pair = len(pair)

rec_report = {}
reconstructions = []

for im1, im2 in pair:
    # Since first row of image pair is always that image itself,
    # so there is cases where im1==im2 which should be avoided.
    if im1 == im2:
        continue

    print(im1, im2)
    if im1 in remaining_images and im2 in remaining_images:
        tracks, p1, p2 = common_track[im1, im2]
        print('selecting pairs...')
        reconstruction, rec_report['bootstrap'] = iba.bootstrap_reconstruction(
            camera_model, exif, ref, track_graph, im1, im2, p1, p2, parameter)
        print('pairs selected...')
        if reconstruction:
            remaining_images.remove(im1)
            remaining_images.remove(im2)
            print('Growing reconstruction ...')
            reconstruction, rec_report['grow'] = iba.grow_reconstruction(
                parameter, track_graph, reconstruction, remaining_images, gcp, exif, ref)
            print('Grow completed')
            reconstructions.append(reconstruction)
            reconstructions = sorted(reconstructions,
                                     key=lambda x: -len(x.shots))
            dataset.save_reconstruction(reconstructions, path[0])
            dataset.save_ply(reconstructions, path[0])
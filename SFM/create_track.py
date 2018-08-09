""" Features to track"""


def load_match(path_output, method_feature):
    path_match = dataset.match_lo(path_output, method_feature)
    match = {}
    for root, dirs, files in os.walk(path_match[0]):
        if len(files) == 0:
            print('No files found in "%s" directory' % (path_match[0]))
            sys.exit('No files found')
            break
        for file_name in files:
            if file_name.endswith(('.npy', '.Npy', '.NPY')):
                im1_match = np.load(os.path.join(
                    path_match[0], file_name)).item()
                im1 = os.path.splitext(file_name)[0] + '.JPG'
                for im2 in im1_match:
                    match[im1, im2] = im1_match[im2]
    return match


def load_features(path_output, method_feature):
    path_feature, _, _, _ = dataset.feature_lo(path_output, method_feature)
    exif_feature = dataset.exif_lo(path_output)
    exif = yaml.safe_load(open(exif_feature[0]))

    _feature = {}
    _nfeature = {}  # Normalised features
    color = {}
    for root, dirs, files in os.walk(path_feature):
        if len(files) == 0:
            print('No files found in "%s" directory' % (path_feature))
            sys.exit('No files found')
            break

        for file_name in files:
            if file_name.endswith(('.npy', '.Npy', '.NPY')):
                im = os.path.splitext(file_name)[0] + '.JPG'
                f, d, c = dataset.unpickle_keypoints(file_name, path_feature)
                pts = dataset.kp2xy(f)
                width = exif[im]['width']
                height = exif[im]['height']
                _nfeature[im] = feature.normalized_image_coordinates(
                    pts, width, height)
                _feature[im] = pts
                color[im] = c

    return _feature, _nfeature, color


import networkx as nx
from src import dataset
from src.dataset import track_lo
from src import track, feature
import os
import sys
import yaml
import argparse
import numpy as np
import time

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

file_para = dataset.para_lo(path_output)
para = yaml.safe_load(open(file_para))

method_feature = para['feature_extractor']

files = track_lo(path_output,method_feature)
path_data = files[1]
#file_track, path_data, path_report, path_logging, path_output


exif_files = dataset.exif_lo(path_output)

# Cheching if exif.json and imagepair.json exists or not
if os.path.isfile(exif_files[0]):
    print('exif.json found')
else:
    sys.exit(
        'No exif.json found. extract_feature.py command first before running matching_feature.py')

# Checking if imagepair.json exist or not
if os.path.isfile(exif_files[1]):
    print('imagepair.json found')
else:
    sys.exit(
        'No imagepair.json found. Run exif.py command first before running matching_feature.py')


exif = yaml.safe_load(open(exif_files[0]))
imagepair = yaml.safe_load(open(exif_files[1]))
no_im = len(exif)


#G = nx.Graph()

# Adding nodes
# for im in range(no_im):
#    im_name = exif[im+1][0] # Since 0 is header file
#    lat = exif[im+1][1]     # Since 0 is header file
#    long = exif[im+1][2]    # Since 0 is header file
#    ele = exif[im+1][3]     # Since 0 is header file
#
#    G.add_node(os.path.basename(im_name),coord=(lat,long,ele),image =im_name )
#    master_im = imagepair[im][0]
#    for pair in range(1,len(imagepair[0])):
#        pair_im = imagepair[im][pair]
#        G.add_edge(master_im,pair_im)
#
#    coord=nx.get_node_attributes(G,'coord')
#
# Adding edges
#nx.draw(G, with_labels=True, font_size=7)

print('loading features ...')
start_time = time.time()
_feature, _nfeature, color = load_features(path_output, method_feature)

end_time = time.time()-start_time
print('features loaded in %s secs' % (end_time))

print('loading matches ...')
match = load_match(path_output, method_feature)
print('matches loaded')

# Minimum track length
min_track_len = para['min_track_length']

# Input given is normalised feature
track_graph, _, tracks, uf = track.create_tracks_graph(
    _nfeature, color, match, min_track_len)
track.save_track_graph(track_graph, path_data)

#  str(image), index(track), feature_id, x, y, r, g, b

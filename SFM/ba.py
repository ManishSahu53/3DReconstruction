""" Incremental bundle adjustment """

from src import track
from src import dataset
from src import incremental_ba as iba
import yaml,os,sys

reload(track)
reload(dataset)
reload(iba)

path_input = './input'
path_output = './output'
method_feature = 'sift'

file = dataset.exif_lo(path_output)  
# Cheching if exif.json and imagepair.json exists or not

if os.path.isfile(file[0]):
    print('exif.json found')
else:
    sys.exit('No exif.json found. extract_feature.py command first before running matching_feature.py')

# Checking if imagepair.json exist or not
if os.path.isfile(file[1]):
    print('imagepair.json found')
else:
    sys.exit('No imagepair.json found. Run exif.py command first before running matching_feature.py')

exif = yaml.safe_load(open(file[0]))
imagepair = yaml.safe_load(open(file[1]))
camera = dataset.load_camera_models(file[2])
no_im = len(exif) 

file_track = dataset.track_lo(path_output,method_feature)

# Loading Track Graph
track_graph = track.load_track_graph(file_track[0])

# Extracting images and tracks
tracks, images = track.tracks_and_images(track_graph)
remaining_images = set(images)

# Taking common features
common_track = track.all_common_tracks(track_graph,tracks)

#  Get good image pair
processes = 16
pair = iba.compute_image_pair(common_track, [exif,camera],processes)
num_pair = len(pair)
for im1, im2 in pairs:
    if im1 in remaining_images and im2 in remaining_images:
        tracks, p1, p2 = common_track[im1, im2]
        reconstruction, rec_report['bootstrap'] = bootstrap_reconstruction(
            data, graph, im1, im2, p1, p2)

        if reconstruction:
            remaining_images.remove(im1)
            remaining_images.remove(im2)
            reconstruction, rec_report['grow'] = grow_reconstruction(
                data, graph, reconstruction, remaining_images, gcp)
            reconstructions.append(reconstruction)
            reconstructions = sorted(reconstructions,
                                     key=lambda x: -len(x.shots))
            data.save_reconstruction(reconstructions)
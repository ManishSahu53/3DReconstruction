""" Features to track"""

def load_match(path_output,method_feature):
    path_match,_,_,_ = dataset.match_lo(path_output,method_feature)
    match = {}
    for root,dirs,files in os.walk(path_match):
        if len(files) == 0:
            print('No files found in "%s" directory'%(path_match))
            sys.exit('No files found')
            break;
        for file_name in files:
            if file_name.endswith(('.npy', '.Npy', '.NPY')):
                im1_match = np.load(os.path.join(path_match,file_name)).item();
                im1 = os.path.splitext(file_name)[0] + '.JPG'
                for im2 in im1_match:
                    match[im1, im2] = im1_match[im2]
    return match

 
def load_features(path_output,method_feature):
    path_feature,_,_,_ = dataset.feature_lo(path_output,method_feature)
    feature = {}
    color = {}
    for root,dirs,files in os.walk(path_feature):
        if len(files) == 0:
            print('No files found in "%s" directory'%(path_feature))
            sys.exit('No files found')
            break;
            
        for file_name in files:
            if file_name.endswith(('.npy', '.Npy', '.NPY')):
                im = os.path.splitext(file_name)[0] + '.JPG' 
                f,d,c = dataset.unpickle_keypoints(file_name,path_feature)
                pts = dataset.kp2xy(f)
                feature[im] = pts
                color[im] = c[0]
                
    return feature, color
        
import networkx as nx
from src import dataset
from src import track
import yaml
import os 
import numpy as np
import sys

image_lo = '/home/indshine-2/Downloads/Dimension/Dimension/test_dataset/Images'
logging_lo = '/home/indshine-2/Downloads/Dimension/Dimension/SFM/extract_feature/output/exif/logging'
path_output = '/home/indshine-2/Downloads/Dimension/Dimension/SFM/extract_feature/output'
method_feature = 'sift'
file_exif,file_imagepair,_,_,_ = dataset.exif_lo(path_output)

exif = yaml.safe_load(open(file_exif))
imagepair = yaml.safe_load(open(file_imagepair))

exif = exif['exif']
no_im = len(exif)-1 # fist is header file
G = nx.Graph()

# Adding nodes
#for im in range(no_im):
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
## Adding edges
#nx.draw(G, with_labels=True, font_size=7)


feature, color = load_features(path_output,method_feature)
match = load_match(path_output,method_feature)
track_graph = track.create_tracks_graph(feature,color,match)
track.save_track_graph(track_graph)
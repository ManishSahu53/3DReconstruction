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
                color[im] = c
                
    return feature, color
        
import networkx as nx
from src import dataset
from src.dataset import checkdir
from src import track
from src.matching import num2method
import os,sys,yaml,argparse
import numpy as np
import time
reload(track)
import matplotlib.pyplot as plt

#parser = argparse.ArgumentParser(description='See description below to see all available options')
#
#parser.add_argument('-i','--input',
#                    help='Output directory feature extraction step.',
#                    required=True)
#                    
#parser.add_argument('-o','--output',
#                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
#                    default = './output/',
#                    required=False)
#
#parser.add_argument('-f','--feature',type=int,
#                    help='feature method that is used for matching'+
#                    ' 1 -- SIFT ' +
#                    ' 2 -- SURF'  +
#                    ' 3 -- ORB'   +
#                    ' 4 -- BRISK' +
#                    ' 5 -- AKAZE' +
#                    ' 6 -- STAR+ BRIEF'+
#                    ' [Default] is SIFT',
#                    default = 1,
#                    required=False)
#                  
#args = parser.parse_args()
#
#path_input = args.input;
#path_output = args.output;
#method_feature = num2method(args.feature); # SIFT,SURF,ORB etc

path_input = './output'
path_output = '/home/indshine-2/Downloads/Dimension/Dimension/SFM/output'
method_feature = 'sift'

# Saving
saving_matches = 'track'

# Update path_output and output directories
path_output = os.path.join(path_output,saving_matches)       
path_logging = os.path.join(path_output,'logging') # Individual file record
path_report = os.path.join(path_output,'report') # Summary of whole file
path_data = os.path.join(path_output,'data',method_feature) # Any data to be saved

# Checking if path  exists, otherwise will be created
checkdir(path_output)
checkdir(path_logging)
checkdir(path_report)
checkdir(path_data)

file_exif,file_imagepair,_,_,_ = dataset.exif_lo(path_input)

   
# Cheching if exif.json and imagepair.json exists or not
if os.path.isfile(file_exif):
    print('exif.json found')
else:
    sys.exit('No exif.json found. extract_feature.py command first before running matching_feature.py')

# Checking if imagepair.json exist or not
if os.path.isfile(file_imagepair):
    print('imagepair.json found')
else:
    sys.exit('No imagepair.json found. Run exif.py command first before running matching_feature.py')
     
    
exif = yaml.safe_load(open(file_exif))
imagepair = yaml.safe_load(open(file_imagepair))

no_im = len(exif) 
#G = nx.Graph()

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

print('loading features ...')
start_time = time.time()
feature, color = load_features(path_input,method_feature)
end_time = time.time()-start_time
print('features loaded in %s secs'%(end_time))

print('loading matches ...')
match = load_match(path_input,method_feature)
print('matches loaded')

track_graph,_,tracks,uf = track.create_tracks_graph(feature,color,match)
track.save_track_graph(track_graph,path_data)

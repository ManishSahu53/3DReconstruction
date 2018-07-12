""" Extracting Exif information from the images"""

# Checking if directory exist
def checkdir(path):
    if path[-1] != '/':
        path = path  + '/'        
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        
# Saving to json format
def tojson(dictA,file_json):
    with open(file_json , 'w') as f:
        json.dump(dictA, f,indent=4, separators=(',', ': '), ensure_ascii=False,encoding='utf-8')     

def eval_frac(value):
    try:
        return float(value.num) / float(value.den)
    except ZeroDivisionError:
        return None
        
def gps_to_decimal(values, reference):
    sign = 1 if reference in 'NE' else -1
    degrees = eval_frac(values[0])
    minutes = eval_frac(values[1])
    seconds = eval_frac(values[2])
    return  sign*(degrees + minutes / 60 + seconds / 3600)
    
# For converting projected to geographic coordinate system
def xy2latlong(easting,northing,zone,hemi):
    lat,long = utm.to_latlon( easting,northing,zone,hemi);
    return lat,long

# For converting geographic to projected coordinate system
def latlong2utm(lat,long):
        coord = utm.from_latlon(lat, long);
        easting =  coord[0];
        northing = coord[1];
        zone = coord[2];
        hemi = coord[3];
        return easting,northing,zone,hemi
        
def distance(lat1,long1,lat2,long2):
    easting1,northing1,zone1,hemi1 = latlong2utm(lat1,long1)
    easting2,northing2,zone2,hemi2 = latlong2utm(lat2,long2)
    return math.sqrt((easting1-easting2)**2 + (northing1-northing2)**2)

def get_neighbour(coord,list_coord,num_neighbour):
    # coord = [ lat[0],long[0],image[0]]
    # list_coord = [lat,long,image]
    neighbour_dist =[];
    sorted_dist = [];
    neighbour_image =[];
    
    for i in range(len(list_coord[0])):
        dist = distance(coord[0],coord[1],list_coord[0][i],list_coord[1][i])
        neighbour_dist.append(dist)
        sorted_dist.append(dist)
    sorted_dist.sort(key=float)
    
    # First element will always be the image itself as distance will be zero
    # with respect to self. So adding [neighbour+1] to get the 
    for j in range(num_neighbour+1):
        neighbour_image.append(list_coord[2][neighbour_dist.index(sorted_dist[j])]);
    return neighbour_image
    
    
       
import exifread
import get_image
import math
#import logging
#reload(logging)
import json
import os
import yaml
import sys
import argparse
import utm

parser = argparse.ArgumentParser(description='See description below to see all available options')

parser.add_argument('-i','--input',
                    help='Input directory containing images. [Default] current directory',
                    required=True)
                    
parser.add_argument('-o','--output',
                    help='Output directory where json file will be stored. [Default] output folder in current directory',
                    default = './output/',
                    required=False)
                    
parser.add_argument('-n','--neighbour',
                    type = int,
                    help='Number of neighbouring images that will be used for matching',
                    default = 9,
                    required=False)
                    
                    
args = parser.parse_args()   
path_image = args.input;
path_output = args.output;
num_neighbour = args.neighbour;


#path_image = '/home/indshine-2/Downloads/Dimension/Data/test/'
#file_json = '/home/indshine-2/Downloads/Dimension/output/SFM/exif/exif.json'

#logging.basicConfig(format='%(asctime)s %(message)s',
#                    filename= path_logging + '/exif.log',
#                    level=logging.DEBUG);
   
# Converting to realative path
path_image = os.path.realpath(path_image)
saving_folder = 'exif'

# Update path_output
path_output = os.path.join(path_output,saving_folder)       
path_logging = os.path.join(path_output,'logging')
path_data = os.path.join(path_output,'data')

# Checking if path  exists, otherwise will be created
checkdir(path_output);
checkdir(path_logging);
checkdir(path_data);

# Fatal error of image directory doesn't exist
if not os.path.exists(os.path.dirname(path_image)):
#    logging.fatal('Input directory given was not found')
    sys.exit('Input directory given was not found')
        
       
def exif(path_image,path_logging):
    list_image_ = get_image.list_image(path_image,path_logging);
    
    # Exit if no image was found
    if len(list_image_) == 0:
    #    logger.fatal('No images were found in input folder')
        sys.exit('No images were found in %s folder'%(path_image))
        
    data=[];
    data.append(["Image","Lat","Long","Elevation","Focal length in mm", "Width(px)","Height(px)","Capture Time"])

#    Looping all over images
    for file_ in list_image_:
        f = open(file_, 'rb');
#        jpeg = pexif.JpegFile.fromFile(file_);
        tags = exifread.process_file(f,details=False)
        ref_lat =  tags['GPS GPSLatitudeRef'].values
        ref_long = tags['GPS GPSLongitudeRef'].values 
        lat = gps_to_decimal(tags["GPS GPSLatitude"].values,ref_lat)
        long= gps_to_decimal(tags["GPS GPSLongitude"].values,ref_long)
        ele = eval_frac(tags["GPS GPSAltitude"].values[0]);
        height = tags["EXIF ExifImageLength"].values[0];
        width = tags["EXIF ExifImageWidth"].values[0];
        capture_time = str(tags["EXIF DateTimeOriginal"])
        focal = float(tags["EXIF FocalLength"].values[0].num)/tags["EXIF FocalLength"].values[0].den
        data.append([os.path.basename(file_),lat,long,ele,focal,width,height,capture_time]);
        
#     Creating a dictionary
    Coordinate = {"Exif": data}
                  
#     Saving dictionary to json file
    tojson(Coordinate,os.path.join(path_data, 'exif.json'))
    print('Extracted Exif details. Data saved to %s'%(os.path.join(path_data,'exif.json')))
    return path_data
    
#     Loading saved json file
#    exif = yaml.safe_load(open(file_json))

def image_pair(path_exif,path_imagepair):
    
#    Loading saved json file
    exif_data = yaml.safe_load(open(os.path.join(path_exif,'exif.json')))
    
#    Initializing variables
    _lat =  [];
    _long = [];
    image = [];
    neighbour = [];
    
#    Extracting coordinates
    
    for i in range(1,len(exif_data["Exif"])):
        _lat.append(exif_data["Exif"][i][1]);
        _long.append(exif_data["Exif"][i][2]);
        image.append(exif_data["Exif"][i][0]);
        
#    Getting neighbouring images    
    for j in range(len(_lat)):
        neighbour.append(get_neighbour([_lat[j],_long[j],image[j]],[_lat,_long,image],num_neighbour))
    
#    Saving data to json format
    tojson(neighbour,os.path.join(path_data, 'imagepair.json'))
    print("Calculated neighbouring images. Data saved to %s"%(os.path.join(path_data, 'imagepair.json')))

        
def main():
    path_data = exif(path_image,path_logging)
    image_pair(path_data,path_output)
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
#        logging.warning('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')
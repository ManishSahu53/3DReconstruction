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

       
import exifread
import get_image
#import logging
#reload(logging)
import json
import os
import yaml
import sys
import argparse

parser = argparse.ArgumentParser(description='See description below to see all available options')

parser.add_argument('-i','--input',
                    help='Input directory containing images. [Default] current directory', 
                    default= './',
                    required=False)
                    
parser.add_argument('-o','--output',
                    help='Output directory where json file will be stored. [Default] output folder in current directory',
                    default = './Output/',
                    required=False)
                    
args = parser.parse_args()                    
                    
path_image = args.input;
file_json = args.output;

path_logging = '/home/indshine-2/Downloads/Dimension/Dimension/logging/'
#path_image = '/home/indshine-2/Downloads/Dimension/Data/test/'
#file_json = '/home/indshine-2/Downloads/Dimension/output/SFM/exif/exif.json'

#logging.basicConfig(format='%(asctime)s %(message)s',
#                    filename= path_logging + '/exif.log',
#                    level=logging.DEBUG);
                    
# Checking output directories
checkdir(file_json);

# Fatal error of image directory doesn't exist
if not os.path.exists(os.path.dirname(path_image)):
#    logging.fatal('Input directory given was not found')
    sys.exit('Input directory given was not found')
        
def exif(path_image,path_logging):
    list_image_ = get_image.list_image(path_image,path_logging);
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
    tojson(Coordinate,file_json + '/exif.json')
    
#     Loading saved json file
#    exif = yaml.safe_load(open(file_json))
    
def main():
    exif(path_image,path_logging)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
#        logging.warning('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')
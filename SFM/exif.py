""" Extracting Exif information from the images"""

import exifread
from src import get_image
import os
import yaml
import sys
import argparse
from src.exif import get_neighbour, gps_to_decimal, eval_frac, tojson, checkdir, reference_coord
from src.dataset import load_parameter

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-i', '--input',
                    help='Input directory containing images. [Default] current directory',
                    required=True)

parser.add_argument('-o', '--output',
                    help='Output directory where json file will be stored. [Default] output folder in current directory',
                    default='./output/',
                    required=False)

parser.add_argument('-n', '--neighbour',
                    type=int,
                    help='Number of neighbouring images that will be used for matching.[Default] is 15',
                    default=15,
                    required=False)

args = parser.parse_args()
path_image = args.input
path_output = args.output
num_neighbour = args.neighbour
path_sensor = './sensor'
file_sensor = os.path.join(path_sensor, 'sensor_data.json')
path_camera = os.path.join(path_image, 'camera.json')
file_parameter = os.path.join(path_image, 'parameter.yaml')

# Loading parameters of the model
parameter = load_parameter(path_image)

#path_image = '/home/indshine-2/Downloads/Dimension/Data/test/'
#file_json = '/home/indshine-2/Downloads/Dimension/output/SFM/exif/exif.json'


# Converting to realative path
path_image = os.path.realpath(path_image)
saving_folder = 'exif'

# Update path_output
path_output = os.path.join(path_output, saving_folder)
path_logging = os.path.join(path_output, 'logging')
path_data = os.path.join(path_output, 'data')

# Checking if path  exists, otherwise will be created
checkdir(path_output)
checkdir(path_logging)
checkdir(path_data)

# Fatal error of image directory doesn't exist
if not os.path.exists(os.path.dirname(path_image)):
    #    logging.fatal('Input directory given was not found')
    sys.exit('Input directory given was not found')

if not os.path.isfile(file_sensor):
    sys.exit('Sensor Width information file at %s path doesnt exist' %
             (path_sensor))

if not os.path.isfile(path_camera):
    sys.exit('Camera model(distortion) information file %s doesnt exist' %
             (path_camera))


def exif(path_image, path_logging):
    list_image_ = get_image.list_image(path_image, path_logging)
    model_camera = yaml.safe_load(open(path_camera))

    # Exit if no image was found
    if len(list_image_) == 0:
        #    logger.fatal('No images were found in input folder')
        sys.exit('No images were found in %s folder' % (path_image))

    data = {}
    sensors_width = yaml.safe_load(open(file_sensor))

#    Looping all over images
    for file_ in list_image_:
        f = open(file_, 'rb')
#        jpeg = pexif.JpegFile.fromFile(file_);
        tags = exifread.process_file(f, details=False)
        camera = tags['Image Model'].values
        ref_lat = tags['GPS GPSLatitudeRef'].values
        ref_long = tags['GPS GPSLongitudeRef'].values
        lat = gps_to_decimal(tags["GPS GPSLatitude"].values, ref_lat)
        long = gps_to_decimal(tags["GPS GPSLongitude"].values, ref_long)
        ele = eval_frac(tags["GPS GPSAltitude"].values[0])
        height = tags["EXIF ExifImageLength"].values[0]
        width = tags["EXIF ExifImageWidth"].values[0]
        capture_time = str(tags["EXIF DateTimeOriginal"])
        focal = float(tags["EXIF FocalLength"].values[0].num) / tags["EXIF FocalLength"].values[0].den

#         Checking for sensor width data
        try:
            sensor_width = sensors_width[camera]
        except KeyError:
            sys.exit(
                'Camera (%s) sensor width information is not present. Add width information then continue' % (camera))

#        Creating a dictionary
        data[os.path.basename(file_)] = {'camera': camera, 'lat': lat,
                                         'long': long, 'ele': ele, 'focal': focal, 'sensor_width': sensor_width,
                                         'width': width, 'height': height, 'time': capture_time}

#     Saving dictionary to json file
    tojson(data, os.path.join(path_data, 'exif.json'))
    tojson(model_camera, os.path.join(path_data, 'camera.json'))
    print('Extracted Exif details. Data saved to %s' %
          (os.path.join(path_data, 'exif.json')))
    return path_data

#     Loading saved json file
#    exif = yaml.safe_load(open(file_json))


def image_pair(path_exif, path_imagepair):

    #    Loading saved json file
    exif_data = yaml.safe_load(open(os.path.join(path_exif, 'exif.json')))

#    Initializing variables
    _lat = []
    _long = []
    image = []
    neighbour = {}

#    Extracting coordinates
    for im in exif_data.keys():
        _lat.append(exif_data[im]['lat'])
        _long.append(exif_data[im]['long'])
        image.append(im)

#     Reference coordinate
    ref = reference_coord(path_exif)

#    Getting neighbouring images
    for im in exif_data.keys():
        neighbour[im] = get_neighbour([exif_data[im]['lat'], exif_data[im]['long'], im], [
                                      _lat, _long, image], num_neighbour)

#    Saving data to json format
    tojson(neighbour, os.path.join(path_data, 'imagepair.json'))
    tojson(ref, os.path.join(path_data, 'reference.json'))
    tojson(parameter, os.path.join(path_data, 'parameter.json'))

    print("Calculated neighbouring images. Data saved to %s" %
          (os.path.join(path_data, 'imagepair.json')))


def main():
    path_data = exif(path_image, path_logging)
    image_pair(path_data, path_output)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        #        logging.warning('Keyboard Interruption')
        sys.exit('Interrupted by keyboard')

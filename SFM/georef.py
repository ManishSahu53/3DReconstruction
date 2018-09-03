""" georeference local coordinates into geographic"""

import os
import logging
import argparse
import yaml

import pyproj
from src import dataset
from src import geo_transformation

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('dataset', help='dataset to process')

parser.add_argument('-i', '--input',
                    help='Input directory containing images.',
                    required=True)

parser.add_argument('-p', '--proj',
                    help='Enter EPSG Code. [Default] is 4326 (WGS 84)',
                    default=4326,
                    required=False)

parser.add_argument('--transformation',
                    help='Print cooordinate transformation matrix',
                    action='store_true',
                    default=False)

parser.add_argument('--image-positions',
                    help='Export image positions',
                    action='store_true',
                    default=False)

parser.add_argument('--reconstruction',
                    help='Export reconstruction.json',
                    action='store_true',
                    default=False)

parser.add_argument('--dense',
                    help='Export dense point cloud (depthmaps/merged.ply)',
                    action='store_true',
                    default=False)

parser.add_argument('-o', '--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
                    default='./output/',
                    required=False)


args = parser.parse_args()
# parsing inputs
path_image = args.input
path_output = args.output
epsg_code = args.proj

# loading parameter file
file_para = dataset.para_lo(path_output)
para = yaml.safe_load(open(file_para))
method_feature = para['feature_extractor']

if not (args.transformation or
        args.image_positions or
        args.reconstruction or
        args.dense):
    print('Nothing to do. At least on of the options should be selected: ')
    print(' --transformation, --image-positions, --reconstruction, --dense')

path_georef = dataset.tgeoref_lo(path_output)
reference = yaml.safe_load(open(dataset.ref_lo(path_output)))

path_reconstruction = dataset.reconstruction_lo(path_output, method_feature)

#    projection = pyproj.Proj(args.proj)
projection = pyproj.Proj(init='epsg:' + str(epsg_code))

transformation = geo_transformation._get_transformation(
                    reference, projection)

if args.transformation:
    output = path_georef[0]
    geo_transformation._write_transformation(transformation, path_output)

if args.image_positions:
    reconstructions = dataset.load_reconstruction(path_reconstruction[0])
    output = path_georef[1]
    geo_transformation._transform_image_positions(reconstructions, transformation,
                                                  output)

if args.reconstruction:
    reconstructions = dataset.load_reconstruction(path_reconstruction[0])
    for r in reconstructions:
        geo_transformation._transform_reconstruction(r, transformation)
    output = path_georef[4]
    dataset.save_reconstruction(reconstructions, output)
    dataset.save_ply(reconstructions, output)
#if args.dense:
#    output = os.path.join(path_georef, 'merged.geocoords.ply')
#    geo_transformation._transform_dense_point_cloud(
#        data, transformation, output)
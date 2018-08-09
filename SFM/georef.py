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
parser.add_argument('-p','--proj',
                    help='Enter EPSG Code',
                    required=True)
parser.add_argument('tr' , '--transformation',
                    help='Print cooordinate transformation matrix',
                    action='store_true',
                    default=False)
parser.add_argument(
    '--image-positions',
    help='Export image positions',
    action='store_true',
    default=False)
parser.add_argument(
    '--reconstruction',
    help='Export reconstruction.json',
    action='store_true',
    default=False)
parser.add_argument(
    '--dense',
    help='Export dense point cloud (depthmaps/merged.ply)',
    action='store_true',
    default=False)

parser.add_argument('-o', '--output',
                    help='Output directory where all the output will be stored. [Default] output folder in current directory',
                    default='./output/',
                    required=False)


def run(args):
    if not (args.transformation or
            args.image_positions or
            args.reconstruction or
            args.dense):
        print('Nothing to do. At least on of the options should be selected: ')
        print(' --transformation, --image-positions, --reconstruction, --dense')

    data = dataset.DataSet(args.dataset)
    reference = yaml.safe_load(open(dataset.ref_lo('./output')))

#    projection = pyproj.Proj(args.proj)
    projection = pyproj.Proj(init='epsg:4326')

    transformation = geo_transformation._get_transformation(
        reference, projection)

    if args.transformation:
        output = args.output or 'geocoords_transformation.txt'
        output_path = os.path.join(data.data_path, output)
        geo_transformation._write_transformation(transformation, output_path)

    if args.image_positions:
        reconstructions = data.load_reconstruction()
        output = args.output or 'image_geocoords.tsv'
        output_path = os.path.join(data.data_path, output)
        geo_transformation._transform_image_positions(reconstructions, transformation,
                                                      output_path)

    if args.reconstruction:
        reconstructions = data.load_reconstruction()
        for r in reconstructions:
            geo_transformation._transform_reconstruction(r, transformation)
        output = args.output or 'reconstruction.geocoords.json'
        data.save_reconstruction(reconstructions, output)

    if args.dense:
        output = args.output or 'depthmaps/merged.geocoords.ply'
        geo_transformation._transform_dense_point_cloud(
            data, transformation, output)

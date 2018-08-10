""" Dataset function which contains location information"""

import os
import numpy as np
import cv2
import json
import io
import sys
import cPickle as pickle
from src import types
from six import iteritems
from src import parameter


def kp2xy(kp):
    point = np.array([(i.pt[0], i.pt[1]) for i in kp])
    return point


def pickle_keypoints(keypoints, descriptors, color):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    return [temp_array, [color]]


def unpickle_keypoints(image, path_feature):
    path_feature = os.path.realpath(path_feature)

    image = os.path.splitext(os.path.basename(image))[0]
    image = os.path.join(path_feature, image+'.npy')
    array = np.load(open(image, "rb"))

    keypoints = []
    descriptors = []
    color = array[1][0]
    array = array[0]

    for point in array:
        temp_feature = cv2.KeyPoint(point[0][0], point[0][1], _size=point[1],
                                    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors), color


# Checking if directory exist
def checkdir(path):
    if path[-1] != '/':
        path = path + '/'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))

# exif file location


def exif_lo(path_output):
    saving_folder = 'exif'
    path_output = os.path.join(path_output, saving_folder)
    path_logging = os.path.join(path_output, 'logging')
    path_data = os.path.join(path_output, 'data')
    file_exif = os.path.join(path_data, 'exif.json')
    file_imagepair = os.path.join(path_data, 'imagepair.json')
    file_camera_model = os.path.join(path_data, 'camera.json')

    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_data)
    return file_exif, file_imagepair, file_camera_model, path_data, path_logging, path_output

def tgeoref_lo(path_output):
    saving_folder = 'georef'
    path_output = os.path.join(path_output, saving_folder)
    path_logging = os.path.join(path_output, 'logging')
    path_data = os.path.join(path_output, 'data')
    file_coord_transf = os.path.join(path_data, 'coord_transformation.json')
    file_image_coord = os.path.join(path_data, 'image_coord.json')
    file_reconstruct_coord = os.path.join(path_data, 'reconstruct_coord_.json')
    file_dense_coord = None
    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_data)
    return file_coord_transf, file_image_coord, file_reconstruct_coord, file_dense_coord, path_data, path_logging, path_output


def para_lo(path_output):
    saving_folder = 'exif'
    path_output = os.path.join(path_output, saving_folder)
    path_data = os.path.join(path_output, 'data')
    file_para = os.path.join(path_data, 'parameter.json')
    return file_para


def ref_lo(path_output):
    saving_folder = 'exif'
    path_output = os.path.join(path_output, saving_folder)
    path_data = os.path.join(path_output, 'data')
    file_ref = os.path.join(path_data, 'reference.json')

    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    return file_ref


def load_parameter(path_input):
    parameter_lo = os.path.join(path_input, 'parameter.yaml')
    return parameter.load_parameter(parameter_lo)

# features location


def feature_lo(path_output, method_feature):
    saving_feature = 'extract_feature'
    # Update path_output and output directories
    path_output = os.path.join(path_output, saving_feature)
    path_logging = os.path.join(path_output, 'logging')
    path_report = os.path.join(path_output, 'report')
    path_data = os.path.join(path_output, 'data',method_feature)

    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return path_data, path_report, path_logging, path_output

# mactchin location


def match_lo(path_output, method_feature):
    saving_matches = 'matching_feature'

    path_output = os.path.join(path_output, saving_matches)
    path_logging = os.path.join(
        path_output, 'logging')  # Individual file record
    path_report = os.path.join(path_output, 'report')  # Summary of whole file
    # Any data to be saved
    path_data = os.path.join(path_output, 'data', method_feature)

    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return path_data, path_report, path_logging, path_output

# track location


def track_lo(path_output, method_feature):
    saving_track = 'track'

    path_output = os.path.join(path_output, saving_track)
    path_logging = os.path.join(
        path_output, 'logging')  # Individual file record
    path_report = os.path.join(path_output, 'report')  # Summary of whole file
    # Any data to be saved
    path_data = os.path.join(path_output, 'data', method_feature)
    file_track = os.path.join(path_data, 'track.csv')

    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return file_track, path_data, path_report, path_logging, path_output

# reconstruction location


def reconstruction_lo(path_output, method_feature):
    saving_feature = 'reconstruction'
    # Update path_output and output directories
    path_output = os.path.join(path_output, saving_feature)
    path_logging = os.path.join(path_output, 'logging')
    path_report = os.path.join(path_output, 'report')
    path_data = os.path.join(path_output, 'data', method_feature)

    # Checking if path  exists, otherwise will be created
    checkdir(path_output)
    checkdir(path_logging)
    checkdir(path_report)
    checkdir(path_data)
    return path_data, path_report, path_logging, path_output

# loading matches


def load_match(file_match):
    match = np.load(open(file_match)).item()
    return match


def tojson(dictA, file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f, indent=4, separators=(',', ': '),
                  ensure_ascii=False, encoding='utf-8')


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

### Cameras ###


def camera_to_json(camera):
    """
    Write camera to a json object
    """
    if camera.projection_type == 'brown':
        return {
            'projection_type': camera.projection_type,
            'width': camera.width,
            'height': camera.height,
            'focal_x': camera.focal_x,
            'focal_y': camera.focal_y,
            'c_x': camera.c_x,
            'c_y': camera.c_y,
            'k1': camera.k1,
            'k2': camera.k2,
            'p1': camera.p1,
            'p2': camera.p2,
            'k3': camera.k3,
            'focal_x_prior': camera.focal_x_prior,
            'focal_y_prior': camera.focal_y_prior,
            'c_x_prior': camera.c_x_prior,
            'c_y_prior': camera.c_y_prior,
            'k1_prior': camera.k1_prior,
            'k2_prior': camera.k2_prior,
            'p1_prior': camera.p1_prior,
            'p2_prior': camera.p2_prior,
            'k3_prior': camera.k3_prior
        }
        raise NotImplementedError


def camera_from_json(key, obj):
    """
    Read camera from a json object
    """
#    Only brownperspective cameras

    camera = types.BrownPerspectiveCamera()
    camera.id = key
    camera.width = obj.get('width', 0)
    camera.height = obj.get('height', 0)
    camera.focal_x = obj['focal_x']
    camera.focal_y = obj['focal_y']
    camera.c_x = obj.get('c_x', 0.0)
    camera.c_y = obj.get('c_y', 0.0)
    camera.k1 = obj.get('k1', 0.0)
    camera.k2 = obj.get('k2', 0.0)
    camera.p1 = obj.get('p1', 0.0)
    camera.p2 = obj.get('p2', 0.0)
    camera.k3 = obj.get('k3', 0.0)
    camera.focal_x_prior = obj.get('focal_x_prior', camera.focal_x)
    camera.focal_y_prior = obj.get('focal_y_prior', camera.focal_y)
    camera.c_x_prior = obj.get('c_x_prior', camera.c_x)
    camera.c_y_prior = obj.get('c_y_prior', camera.c_y)
    camera.k1_prior = obj.get('k1_prior', camera.k1)
    camera.k2_prior = obj.get('k2_prior', camera.k2)
    camera.p1_prior = obj.get('p1_prior', camera.k1)
    camera.p2_prior = obj.get('p2_prior', camera.k2)
    camera.k3_prior = obj.get('k3_prior', camera.k1)
    return camera


def cameras_from_json(obj):
    """
    Read cameras from a json object
    """
    cameras = {}
    for key, value in iteritems(obj):
        cameras[key] = camera_from_json(key, value)
    return cameras


def load_camera_models(file_camera):
    obj = json.load(open(file_camera))
    return cameras_from_json(obj)


# Shot to json
def shot_to_json(shot):
    """
    Write shot to a json object
    """
    obj = {
        'rotation': list(shot.pose.rotation),
        'translation': list(shot.pose.translation),
        'camera': shot.camera.id
    }
    if shot.metadata is not None:
        if shot.metadata.orientation is not None:
            obj['orientation'] = shot.metadata.orientation
        if shot.metadata.capture_time is not None:
            obj['capture_time'] = shot.metadata.capture_time
        if shot.metadata.gps_dop is not None:
            obj['gps_dop'] = shot.metadata.gps_dop
        if shot.metadata.gps_position is not None:
            obj['gps_position'] = shot.metadata.gps_position
        if shot.metadata.accelerometer is not None:
            obj['accelerometer'] = shot.metadata.accelerometer
        if shot.metadata.compass is not None:
            obj['compass'] = shot.metadata.compass
        if shot.metadata.skey is not None:
            obj['skey'] = shot.metadata.skey
    if shot.mesh is not None:
        obj['vertices'] = shot.mesh.vertices
        obj['faces'] = shot.mesh.faces
    if hasattr(shot, 'scale'):
        obj['scale'] = shot.scale
    if hasattr(shot, 'covariance'):
        obj['covariance'] = shot.covariance.tolist()
    if hasattr(shot, 'merge_cc'):
        obj['merge_cc'] = shot.merge_cc
    return obj


def shot_from_json(key, obj, cameras):
    """
    Read shot from a json object
    """
    pose = types.Pose()
    pose.rotation = obj["rotation"]
    if "translation" in obj:
        pose.translation = obj["translation"]

    metadata = types.ShotMetadata()
    metadata.orientation = obj.get("orientation")
    metadata.capture_time = obj.get("capture_time")
    metadata.gps_dop = obj.get("gps_dop")
    metadata.gps_position = obj.get("gps_position")

    shot = types.Shot()
    shot.id = key
    shot.metadata = metadata
    shot.pose = pose
    shot.camera = cameras.get(obj["camera"])

    if 'scale' in obj:
        shot.scale = obj['scale']
    if 'covariance' in obj:
        shot.covariance = np.array(obj['covariance'])
    if 'merge_cc' in obj:
        shot.merge_cc = obj['merge_cc']
    if 'vertices' in obj and 'faces' in obj:
        shot.mesh = types.ShotMesh()
        shot.mesh.vertices = obj['vertices']
        shot.mesh.faces = obj['faces']

    return shot


def point_from_json(key, obj):
    """
    Read a point from a json object
    """
    point = types.Point()
    point.id = key
    point.color = obj["color"]
    point.coordinates = obj["coordinates"]
    if "reprojection_error" in obj:
        point.reprojection_error = obj["reprojection_error"]
    return point


def point_to_json(point):
    """
    Write a point to a json object
    """
    return {
        'color': list(point.color),
        'coordinates': list(point.coordinates),
        'reprojection_error': point.reprojection_error
    }

# Reconstructions from/to json


def reconstruction_from_json(obj):
    """
    Read a reconstruction from a json object
    """
    reconstruction = types.Reconstruction()

    # Extract cameras
    for key, value in iteritems(obj['cameras']):
        camera = camera_from_json(key, value)
        reconstruction.add_camera(camera)

    # Extract shots
    for key, value in iteritems(obj['shots']):
        shot = shot_from_json(key, value, reconstruction.cameras)
        reconstruction.add_shot(shot)

    # Extract points
    if 'points' in obj:
        for key, value in iteritems(obj['points']):
            point = point_from_json(key, value)
            reconstruction.add_point(point)

    # Extract pano_shots
    if 'pano_shots' in obj:
        reconstruction.pano_shots = {}
        for key, value in iteritems(obj['pano_shots']):
            shot = shot_from_json(key, value, reconstruction.cameras)
            reconstruction.pano_shots[shot.id] = shot

    # Extract main and unit shots
    if 'main_shot' in obj:
        reconstruction.main_shot = obj['main_shot']
    if 'unit_shot' in obj:
        reconstruction.unit_shot = obj['unit_shot']

    return reconstruction


def reconstructions_from_json(obj):
    """
    Read all reconstructions from a json object
    """
    return [reconstruction_from_json(i) for i in obj]


def reconstruction_to_json(reconstruction):
    """
    Write a reconstruction to a json object
    """
    obj = {
        "cameras": {},
        "shots": {},
        "points": {}
    }

    # Extract cameras
    for camera in reconstruction.cameras.values():
        obj['cameras'][camera.id] = camera_to_json(camera)

    # Extract shots
    for shot in reconstruction.shots.values():
        obj['shots'][shot.id] = shot_to_json(shot)

    # Extract points
    for point in reconstruction.points.values():
        obj['points'][point.id] = point_to_json(point)

    # Extract pano_shots
    if hasattr(reconstruction, 'pano_shots'):
        obj['pano_shots'] = {}
        for shot in reconstruction.pano_shots.values():
            obj['pano_shots'][shot.id] = shot_to_json(shot)

    # Extract main and unit shots
    if hasattr(reconstruction, 'main_shot'):
        obj['main_shot'] = reconstruction.main_shot
    if hasattr(reconstruction, 'unit_shot'):
        obj['unit_shot'] = reconstruction.unit_shot

    return obj


def reconstructions_to_json(reconstructions):
    """
    Write all reconstructions to a json object
    """
    return [reconstruction_to_json(i) for i in reconstructions]


def save_reconstruction(reconstruction, path_output, minify=False):
    reconstruct = reconstructions_to_json(reconstruction)
    with io.open(os.path.join(path_output, 'reconstruction.json'), 'w', encoding='utf-8') as fout:
        json_dump(reconstruct, fout, minify)
    return reconstruct

def load_reconstruction(path_output):
    with open(os.path.join(path_output, 'reconstruction.json')) as fin:
        reconstructions = reconstructions_from_json(json.load(fin))
    return reconstructions

# JSON Dump


def _json_dump_python_2_pached(obj, fp, skipkeys=False, ensure_ascii=True, check_circular=True,
                               allow_nan=True, cls=None, indent=None, separators=None,
                               encoding='utf-8', default=None, sort_keys=False, **kw):
    """Serialize ``obj`` as a JSON formatted stream to ``fp`` (a
    ``.write()``-supporting file-like object).

    If ``skipkeys`` is true then ``dict`` keys that are not basic types
    (``str``, ``unicode``, ``int``, ``long``, ``float``, ``bool``, ``None``)
    will be skipped instead of raising a ``TypeError``.

    If ``ensure_ascii`` is true (the default), all non-ASCII characters in the
    output are escaped with ``\\uXXXX`` sequences, and the result is a ``str``
    instance consisting of ASCII characters only.  If ``ensure_ascii`` is
    ``False``, some chunks written to ``fp`` may be ``unicode`` instances.
    This usually happens because the input contains unicode strings or the
    ``encoding`` parameter is used. Unless ``fp.write()`` explicitly
    understands ``unicode`` (as in ``codecs.getwriter``) this is likely to
    cause an error.

    If ``check_circular`` is false, then the circular reference check
    for container types will be skipped and a circular reference will
    result in an ``OverflowError`` (or worse).

    If ``allow_nan`` is false, then it will be a ``ValueError`` to
    serialize out of range ``float`` values (``nan``, ``inf``, ``-inf``)
    in strict compliance of the JSON specification, instead of using the
    JavaScript equivalents (``NaN``, ``Infinity``, ``-Infinity``).

    If ``indent`` is a non-negative integer, then JSON array elements and
    object members will be pretty-printed with that indent level. An indent
    level of 0 will only insert newlines. ``None`` is the most compact
    representation.  Since the default item separator is ``', '``,  the
    output might include trailing whitespace when ``indent`` is specified.
    You can use ``separators=(',', ': ')`` to avoid this.

    If ``separators`` is an ``(item_separator, dict_separator)`` tuple
    then it will be used instead of the default ``(', ', ': ')`` separators.
    ``(',', ':')`` is the most compact JSON representation.

    ``encoding`` is the character encoding for str instances, default is UTF-8.

    ``default(obj)`` is a function that should return a serializable version
    of obj or raise TypeError. The default simply raises TypeError.

    If *sort_keys* is ``True`` (default: ``False``), then the output of
    dictionaries will be sorted by key.

    To use a custom ``JSONEncoder`` subclass (e.g. one that overrides the
    ``.default()`` method to serialize additional types), specify it with
    the ``cls`` kwarg; otherwise ``JSONEncoder`` is used.

    """
    # cached encoder
    if (not skipkeys and ensure_ascii and
        check_circular and allow_nan and
        cls is None and indent is None and separators is None and
            encoding == 'utf-8' and default is None and not sort_keys and not kw):
        iterable = json._default_encoder.iterencode(obj)
    else:
        if cls is None:
            cls = json.JSONEncoder
        iterable = cls(skipkeys=skipkeys, ensure_ascii=ensure_ascii,
                       check_circular=check_circular, allow_nan=allow_nan, indent=indent,
                       separators=separators, encoding=encoding,
                       default=default, sort_keys=sort_keys, **kw).iterencode(obj)
    # could accelerate with writelines in some versions of Python, at
    # a debuggability cost
    for chunk in iterable:
        fp.write(unicode(chunk))  # Convert chunks to unicode before writing


def json_dump_kwargs(minify=False):
    if minify:
        indent, separators = None, (',', ':')
    else:
        indent, separators = 4, None
    return dict(indent=indent, ensure_ascii=False,
                separators=separators)


def json_dump(data, fout, minify=False):
    kwargs = json_dump_kwargs(minify)
    if sys.version_info >= (3, 0):
        return json.dump(data, fout, **kwargs)
    else:
        # Python 2 json decoders can unpredictably return str or unicode
        # We use a patched json.dump function to always convert to unicode
        # See https://bugs.python.org/issue13769
        return _json_dump_python_2_pached(data, fout, **kwargs)


def json_dumps(data, minify=False):
    kwargs = json_dump_kwargs(minify)
    if sys.version_info >= (3, 0):
        return json.dumps(data, **kwargs)
    else:
        # Python 2 json decoders can unpredictably return str or unicode.
        # We use always convert to unicode
        # See https://bugs.python.org/issue13769
        return unicode(json.dumps(data, **kwargs))


# PLY

def reconstruction_to_ply(reconstructions, no_cameras=False, no_points=False):
    """Export reconstruction points as a PLY string."""
    vertices = []

    for i in range(len(reconstructions)):
        reconstruction = reconstructions[i]

        if not no_points:
            for point in reconstruction.points.values():
                p, c = point.coordinates, point.color
                s = "{} {} {} {} {} {}".format(
                    p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2]))
                vertices.append(s)

        if not no_cameras:
            for shot in reconstruction.shots.values():
                o = shot.pose.get_origin()
                R = shot.pose.get_rotation_matrix()
                for axis in range(3):
                    c = 255 * np.eye(3)[axis]
                    for depth in np.linspace(0, 1, 10):
                        p = o + depth * R[axis]
                        s = "{} {} {} {} {} {}".format(
                            p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2]))
                        vertices.append(s)

    header = [
        u"ply",
        u"format ascii 1.0",
        u"element vertex {}".format(len(vertices)),
        u"property float x",
        u"property float y",
        u"property float z",
        u"property uchar diffuse_red",
        u"property uchar diffuse_green",
        u"property uchar diffuse_blue",
        u"end_header",
    ]

    return '\n'.join(header + vertices + [''])


def save_ply(reconstruction, path_output, no_cameras=False, no_points=False):
    """Save a reconstruction in PLY format."""
    ply = reconstruction_to_ply(reconstruction, no_cameras, no_points)
    with io.open(os.path.join(path_output, 'reconstruction.ply'), 'w', encoding='utf-8') as fout:
        fout.write(ply)

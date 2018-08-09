import os
import numpy as np
from src import geo
import pyproj
import io
from src.feature import normalized_image_coordinates
from src import types

def _get_transformation(reference, projection):
        """Get the linear transform from reconstruction coords to geocoords."""
        p = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]]
        q = [_transform(point, reference, projection) for point in p]

        transformation = np.array([
            [q[0][0] - q[3][0], q[1][0] - q[3][0], q[2][0] - q[3][0], q[3][0]],
            [q[0][1] - q[3][1], q[1][1] - q[3][1], q[2][1] - q[3][1], q[3][1]],
            [q[0][2] - q[3][2], q[1][2] - q[3][2], q[2][2] - q[3][2], q[3][2]],
            [0, 0, 0, 1]
        ])
        return transformation

def _write_transformation(transformation, filename):

    """Write the 4x4 matrix transformation to a text file."""
    with io.open(filename, 'w', encoding='utf-8') as fout:
        for row in transformation:
            fout.write(' '.join(map(str, row)))
            fout.write('\n')

def _transform(point, reference, projection):
    """Transform on point from local coords to a proj4 projection."""
    lat, lon, altitude = geo.lla_from_topocentric(
        point[0], point[1], point[2],
        reference['lat'], reference['long'], reference['ele'])
    easting, northing = projection(lon, lat)
    return [easting, northing, altitude]

def _transform_image_positions(reconstructions, transformation, output):
    A, b = transformation[:3, :3], transformation[:3, 3]

    rows = ['Image\tX\tY\tZ']
    for r in reconstructions:
        for shot in r.shots.values():
            o = shot.pose.get_origin()
            to = np.dot(A, o) + b
            row = [shot.id, to[0], to[1], to[2]]
            rows.append('\t'.join(map(str, row)))

    text = '\n'.join(rows + [''])
    with open(output, 'w') as fout:
        fout.write(text)

def _transform_reconstruction(reconstruction, transformation):
    """Apply a transformation to a reconstruction in-place."""
    A, b = transformation[:3, :3], transformation[:3, 3]
    A1 = np.linalg.inv(A)
    b1 = -np.dot(A1, b)

    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        t = shot.pose.translation
        shot.pose.set_rotation_matrix(np.dot(R, A1))
        shot.pose.translation = list(np.dot(R, b1) + t)

    for point in reconstruction.points.values():
        point.coordinates = list(np.dot(A, point.coordinates) + b)

def _transform_dense_point_cloud(data, transformation, output):
    """Apply a transformation to the merged point cloud."""
    A, b = transformation[:3, :3], transformation[:3, 3]
    input_path = os.path.join(data._depthmap_path(), 'merged.ply')
    output_path = os.path.join(data.data_path, output)
    with io.open(input_path, 'r', encoding='utf-8') as fin:
        with io.open(output_path, 'r', encoding='utf-8') as fout:
            for i, line in enumerate(fin):
                if i < 13:
                    fout.write(line)
                else:
                    x, y, z, nx, ny, nz, red, green, blue = line.split()
                    x, y, z = np.dot(A, map(float, [x, y, z])) + b
                    nx, ny, nz = np.dot(A, map(float, [nx, ny, nz]))
                    fout.write(
                        "{} {} {} {} {} {} {} {} {}\n".format(
                            x, y, z, nx, ny, nz, red, green, blue))


def _read_ground_control_points_list_line(line, projection, reference_lla, exif):
    words = line.split()
    easting, northing, alt, pixel_x, pixel_y = map(float, words[:5])
    shot_id = words[5]

    # Convert 3D coordinates
    if projection is not None:
        lon, lat = projection(easting, northing, inverse=True)
    else:
        lon, lat = easting, northing
    x, y, z = geo.topocentric_from_lla(
        lat, lon, alt,
        reference_lla['latitude'],
        reference_lla['longitude'],
        reference_lla['altitude'])

    # Convert 2D coordinates
    d = exif[shot_id]
    coordinates = normalized_image_coordinates(
        np.array([[pixel_x, pixel_y]]), d['width'], d['height'])[0]

    o = types.GroundControlPointObservation()
    o.lla = np.array([lat, lon, alt])
    o.coordinates = np.array([x, y, z])
    o.shot_id = shot_id
    o.shot_coordinates = coordinates
    return o


def _parse_utm_projection_string(line):
    """Convert strings like 'WGS84 UTM 32N' to a proj4 definition."""
    words = line.lower().split()
    assert len(words) == 3
    zone = line.split()[2].upper()
    if zone[-1] == 'N':
        zone_number = int(zone[:-1])
        zone_hemisphere = 'north'
    elif zone[-1] == 'S':
        zone_number = int(zone['-1'])
        zone_hemisphere = 'south'
    else:
        zone_number = int(zone)
        zone_hemisphere = 'north'
    s = '+proj=utm +zone={} +{} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    return s.format(zone_number, zone_hemisphere)


def _parse_projection(line):
    """Build a proj4 from the GCP format line."""
    if line.strip() == 'WGS84':
        return None
    elif line.upper().startswith('WGS84 UTM'):
        return pyproj.Proj(_parse_utm_projection_string(line))
    elif '+proj' in line:
        return pyproj.Proj(line)
    else:
        raise ValueError("Un-supported geo system definition: {}".format(line))


def read_ground_control_points_list(fileobj, reference_lla, exif):
    """Read a ground control point list file.

    It requires the points to be in the WGS84 lat, lon, alt format.
    """
    lines = fileobj.readlines()
    projection = _parse_projection(lines[0])
    points = [_read_ground_control_points_list_line(line, projection, reference_lla, exif)
              for line in lines[1:]]
    return points


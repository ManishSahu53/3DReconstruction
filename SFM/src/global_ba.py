""" Incremental bundle adjustment """
import pyopengv
import cv2
import numpy as np
from src import context
from src import types
from src import geo
from src import multiview
from src.feature import normalized_image_coordinates, denormalized_image_coordinates

from collections import defaultdict
from src import transformations as tf

from six import iteritems
import six
import csfm
import time
from src.track import tracks_and_images
from src import dataset
from scipy.sparse import lil_matrix


class TrackTriangulator:
    """Triangulate tracks in a reconstruction.

    Caches shot origin and rotation matrix
    """

    def __init__(self, graph, reconstruction):
        """Build a triangulator for a specific reconstruction."""
        self.graph = graph
        self.reconstruction = reconstruction
        self.origins = {}
        self.rotation_inverses = {}
        self.Rts = {}

    def triangulate(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track and add point to reconstruction."""
        os, bs = [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                os.append(self._shot_origin(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                r = self._shot_rotation_inverse(shot)
                bs.append(r.dot(b))

        if len(os) >= 2:
            e, X = csfm.triangulate_bearings_midpoint(
                os, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
            if X is not None:
                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()
                self.reconstruction.add_point(point)
    
    def paint_reconstruction(self, graph, reconstruction):
        """Set the color of the points from the color of the tracks."""
        for k, point in iteritems(reconstruction.points):
            point.color = six.next(six.itervalues(graph[k]))['feature_color']
        
    def triangulate_dlt(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track using DLT and add point to reconstruction."""
        Rts, bs = [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                Rts.append(self._shot_Rt(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                bs.append(b)

        if len(Rts) >= 2:
            e, X = csfm.triangulate_bearings_dlt(
                Rts, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
            if X is not None:
                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()
                self.reconstruction.add_point(point)

    def _shot_origin(self, shot):
        if shot.id in self.origins:
            return self.origins[shot.id]
        else:
            o = shot.pose.get_origin()
            self.origins[shot.id] = o
            return o

    def _shot_rotation_inverse(self, shot):
        if shot.id in self.rotation_inverses:
            return self.rotation_inverses[shot.id]
        else:
            r = shot.pose.get_rotation_matrix().T
            self.rotation_inverses[shot.id] = r
            return r

    def _shot_Rt(self, shot):
        if shot.id in self.Rts:
            return self.Rts[shot.id]
        else:
            r = shot.pose.get_Rt()
            self.Rts[shot.id] = r
            return r


def get_image_metadata(exif, ref, image):
    """Get image metadata as a ShotMetadata object."""

    metadata = types.ShotMetadata()
    reflla = ref

    lat = exif[image]['lat']
    lon = exif[image]['long']
    ele = exif[image]['ele']
#    x, y, z = geo.ecef_from_lla(lat, lon, ele)
    x, y, z = geo.topocentric_from_lla(
        lat, lon, ele,
        reflla['lat'], reflla['long'], reflla['ele'])
    metadata.gps_position = [x, y, z]

    # Getting degree of precision
    metadata.gps_dop = exif.get('dop', 15.0)

    metadata.orientation = exif.get('orientation', 1)

    if 'accelerometer' in exif:
        metadata.accelerometer = exif['accelerometer']

    if 'compass' in exif:
        metadata.compass = exif['compass']

    if 'capture_time' in exif:
        metadata.capture_time = exif['capture_time']

    if 'skey' in exif:
        metadata.skey = exif['skey']

    return metadata


def compute_image_pair(track_dict, list_, para):  # list = [exif,camera_model]
    """All matched image pairs sorted by reconstructability."""
    args = _pair_reconstructability_arguments(
        track_dict, list_, para)  # list = [exif,camera_model]
    result = context.parallel_map(
        _compute_pair_reconstructability, args, para['processes'])
    result = list(result)
    pairs = [(im1, im2) for im1, im2, r in result if r > 0]
    score = [r for im1, im2, r in result if r > 0]
    order = np.argsort(-np.array(score))
    return [pairs[o] for o in order]


def triangulate_shot_features(graph, reconstruction, shot_id, reproj_threshold,
                              min_ray_angle):
    """Reconstruct as many tracks seen in shot_id as possible."""
    triangulator = TrackTriangulator(graph, reconstruction)

    for track in graph[shot_id]:
        if track not in reconstruction.points:
            triangulator.triangulate(track, reproj_threshold, min_ray_angle)


# list = [exif,camera_model]
def _pair_reconstructability_arguments(track_dict, list, para):
    # Outlier threshold (in pixels) for essential matrix estimation
    threshold = 4 * para['five_point_algo_threshold']
    cameras = list[1]
    exif = list[0]
    args = []
    for (im1, im2), (tracks, p1, p2) in iteritems(track_dict):
        camera1 = cameras[exif[im1]['camera']]
        camera2 = cameras[exif[im2]['camera']]
        args.append((im1, im2, p1, p2, camera1, camera2, threshold))
    return args


def _compute_pair_reconstructability(args):
    im1, im2, p1, p2, camera1, camera2, threshold = args
    R, inliers = two_view_reconstruction_rotation_only(
        p1, p2, camera1, camera2, threshold)
    r = pairwise_reconstructability(len(p1), len(inliers))
    return (im1, im2, r)


def two_view_reconstruction_rotation_only(p1, p2, camera1, camera2, threshold):
    """Find rotation between two views from point correspondences.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    R = pyopengv.relative_pose_ransac_rotation_only(
        b1, b2, 1 - np.cos(threshold), 1000)
    inliers = _two_view_rotation_inliers(b1, b2, R, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), inliers


def _two_view_rotation_inliers(b1, b2, R, threshold):
    br2 = R.dot(b2.T).T
    ok = np.linalg.norm(br2 - b1, axis=1) < threshold
    return np.nonzero(ok)[0]


def pairwise_reconstructability(common_tracks, rotation_inliers):
    """Likeliness of an image pair giving a good initial reconstruction."""
    outliers = common_tracks - rotation_inliers
    outlier_ratio = float(outliers) / common_tracks
    if outlier_ratio >= 0.3:
        return outliers
    else:
        return 0


def _get_camera_from_bundle(ba, camera):
    """Read camera parameters from a bundle adjustment problem."""
    if camera.projection_type == 'perspective':
        c = ba.get_perspective_camera(str(camera.id))
        camera.focal = c.focal
        camera.k1 = c.k1
        camera.k2 = c.k2
    elif camera.projection_type == 'brown':
        c = ba.get_brown_perspective_camera(str(camera.id))
        camera.focal_x = c.focal_x
        camera.focal_y = c.focal_y
        camera.c_x = c.c_x
        camera.c_y = c.c_y
        camera.k1 = c.k1
        camera.k2 = c.k2
        camera.p1 = c.p1
        camera.p2 = c.p2
        camera.k3 = c.k3
    elif camera.projection_type == 'fisheye':
        c = ba.get_fisheye_camera(str(camera.id))
        camera.focal = c.focal
        camera.k1 = c.k1
        camera.k2 = c.k2


def _add_camera_to_bundle(ba, camera, constant):
    """Add camera to a bundle adjustment problem."""
    if camera.projection_type == 'perspective':
        ba.add_perspective_camera(
            str(camera.id), camera.focal, camera.k1, camera.k2,
            camera.focal_prior, camera.k1_prior, camera.k2_prior,
            constant)
    elif camera.projection_type == 'brown':
        c = csfm.BABrownPerspectiveCamera()
        c.id = str(camera.id)
        c.focal_x = camera.focal_x
        c.focal_y = camera.focal_y
        c.c_x = camera.c_x
        c.c_y = camera.c_y
        c.k1 = camera.k1
        c.k2 = camera.k2
        c.p1 = camera.p1
        c.p2 = camera.p2
        c.k3 = camera.k3
        c.focal_x_prior = camera.focal_x_prior
        c.focal_y_prior = camera.focal_y_prior
        c.c_x_prior = camera.c_x_prior
        c.c_y_prior = camera.c_y_prior
        c.k1_prior = camera.k1_prior
        c.k2_prior = camera.k2_prior
        c.p1_prior = camera.p1_prior
        c.p2_prior = camera.p2_prior
        c.k3_prior = camera.k3_prior
        c.constant = constant
        ba.add_brown_perspective_camera(c)


def resect(exif, graph, reconstruction, shot_id, ref, para):
    """Try resecting and adding a shot to the reconstruction.

    Return:
        True on success.
    """
    exif_im = exif[shot_id]
    camera = reconstruction.cameras[exif_im['camera']]

    bs = []
    Xs = []
    for track in graph[shot_id]:
        if track in reconstruction.points:
            x = graph[track][shot_id]['feature']
            b = camera.pixel_bearing(x)
            bs.append(b)
            Xs.append(reconstruction.points[track].coordinates)
    bs = np.array(bs)
    Xs = np.array(Xs)
    if len(bs) < 5:
        return False, {'num_common_points': len(bs)}

    threshold = para['resection_threshold']
    T = pyopengv.absolute_pose_ransac(
        bs, Xs, "KNEIP", 1 - np.cos(threshold), 1000)

    R = T[:, :3]
    t = T[:, 3]

    reprojected_bs = R.T.dot((Xs - t).T).T
    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

    inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
    ninliers = int(sum(inliers))

#    logger.info("{} resection inliers: {} / {}".format(
#        shot_id, ninliers, len(bs)))
    report = {
        'num_common_points': len(bs),
        'num_inliers': ninliers,
    }
    if ninliers >= para['resection_min_inliers']:
        R = T[:, :3].T
        t = -R.dot(T[:, 3])
        shot = types.Shot()
        shot.id = shot_id
        shot.camera = camera
        shot.pose = types.Pose()
        shot.pose.set_rotation_matrix(R)
        shot.pose.translation = t
        shot.metadata = get_image_metadata(exif, ref, shot_id)
        reconstruction.add_shot(shot)
        bundle_single_view(graph, reconstruction, shot_id, para)
        return True, report
    else:
        return False, report


def shot_direct_neighbors(graph, reconstruction, shot_id):
    """Reconstructed shots sharing reconstructed points with a given shot."""
    neighbors = set()
    for track_id in graph[shot_id]:
        if track_id in reconstruction.points:
            for neighbor in graph[track_id]:
                if neighbor in reconstruction.shots:
                    neighbors.add(neighbor)
    return neighbors


def reconstructed_points_for_images(graph, reconstruction, images):
    """Number of reconstructed points visible on each image.

    Returns:
        A list of (image, num_point) pairs sorted by decreasing number
        of points.
    """
    res = []
    for image in images:
        if image not in reconstruction.shots:
            common_tracks = 0
            for track in graph[image]:
                if track in reconstruction.points:
                    common_tracks += 1
            res.append((image, common_tracks))
    return sorted(res, key=lambda x: -x[1])


def shot_neighborhood(graph, reconstruction, central_shot_id, radius):
    """Reconstructed shots near a given shot.

    Returns:
        a tuple with interior and boundary:
        - interior: the list of shots at distance smaller than radius
        - boundary: shots at distance radius

    Central shot is at distance 0.  Shots at distance n + 1 share at least
    one point with shots at distance n.
    """
    interior = set()
    boundary = set([central_shot_id])
    for distance in range(radius):
        new_boundary = set()
        for shot_id in boundary:
            neighbors = shot_direct_neighbors(graph, reconstruction, shot_id)
            for neighbor in neighbors:
                if neighbor not in boundary and neighbor not in interior:
                    new_boundary.add(neighbor)
        interior.update(boundary)
        boundary = new_boundary
    return interior, boundary


def paint_reconstruction(para, graph, reconstruction):
    """Set the color of the points from the color of the tracks."""
    for k, point in iteritems(reconstruction.points):
        point.color = six.next(six.itervalues(graph[k]))['feature_color']


class ShouldBundle:
    """Helper to keep track of when to run bundle."""

    def __init__(self, para, reconstruction):
        self.interval = para['bundle_interval']
        self.new_points_ratio = para['bundle_new_points_ratio']
        self.done(reconstruction)

    def should(self, reconstruction):
        max_points = self.num_points_last * self.new_points_ratio
        max_shots = self.num_shots_last + self.interval
        return (len(reconstruction.points) >= max_points or
                len(reconstruction.shots) >= max_shots)

    def done(self, reconstruction):
        self.num_points_last = len(reconstruction.points)
        self.num_shots_last = len(reconstruction.shots)


class ShouldRetriangulate:
    """Helper to keep track of when to re-triangulate."""

    def __init__(self, para, reconstruction):
        self.active = para['retriangulation']
        self.ratio = para['retriangulation_ratio']
        self.done(reconstruction)

    def should(self, reconstruction):
        max_points = self.num_points_last * self.ratio
        return self.active and len(reconstruction.points) > max_points

    def done(self, reconstruction):
        self.num_points_last = len(reconstruction.points)


def bundle(graph, reconstruction, gcp, para):
    """Bundle adjust a reconstruction."""
    fix_cameras = not para['optimize_camera_parameters']

    ba = csfm.BundleAdjuster()

    for camera in reconstruction.cameras.values():
        _add_camera_to_bundle(ba, camera, fix_cameras)

    for shot in reconstruction.shots.values():
        r = shot.pose.rotation
        t = shot.pose.translation
        ba.add_shot(
            str(shot.id), str(shot.camera.id),
            r[0], r[1], r[2],
            t[0], t[1], t[2],
            False
        )

    for point in reconstruction.points.values():
        x = point.coordinates
        ba.add_point(str(point.id), x[0], x[1], x[2], False)

    for shot_id in reconstruction.shots:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    ba.add_observation(str(shot_id), str(track),
                                       *graph[shot_id][track]['feature'])

    if para['bundle_use_gps']:
        for shot in reconstruction.shots.values():
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

    if para['bundle_use_gcp'] and gcp:
        for observation in gcp:
            if observation.shot_id in reconstruction.shots:
                ba.add_ground_control_point_observation(
                    str(observation.shot_id),
                    observation.coordinates[0],
                    observation.coordinates[1],
                    observation.coordinates[2],
                    observation.shot_coordinates[0],
                    observation.shot_coordinates[1])

    ba.set_loss_function(para['loss_function'],
                         para['loss_function_threshold'])
    ba.set_reprojection_error_sd(para['reprojection_error_sd'])
    ba.set_internal_parameters_prior_sd(
        para['exif_focal_sd'],
        para['principal_point_sd'],
        para['radial_distorsion_k1_sd'],
        para['radial_distorsion_k2_sd'],
        para['radial_distorsion_p1_sd'],
        para['radial_distorsion_p2_sd'],
        para['radial_distorsion_k3_sd'])
    ba.set_num_threads(para['processes'])

    ba.run()

    for camera in reconstruction.cameras.values():
        _get_camera_from_bundle(ba, camera)

    for shot in reconstruction.shots.values():
        s = ba.get_shot(str(shot.id))
        shot.pose.rotation = [s.rx, s.ry, s.rz]
        shot.pose.translation = [s.tx, s.ty, s.tz]

    for point in reconstruction.points.values():
        p = ba.get_point(str(point.id))
        point.coordinates = [p.x, p.y, p.z]
        point.reprojection_error = p.reprojection_error

    report = {
        'brief_report': ba.brief_report(),
    }
    return report


def bundle_single_view(graph, reconstruction, shot_id, para):
    """Bundle adjust a single camera."""
    ba = csfm.BundleAdjuster()
    shot = reconstruction.shots[shot_id]
    camera = shot.camera

    _add_camera_to_bundle(ba, camera, constant=True)

    r = shot.pose.rotation
    t = shot.pose.translation
    ba.add_shot(
        str(shot.id), str(camera.id),
        r[0], r[1], r[2],
        t[0], t[1], t[2],
        False
    )

    for track_id in graph[shot_id]:
        if track_id in reconstruction.points:
            track = reconstruction.points[track_id]
            x = track.coordinates
            ba.add_point(str(track_id), x[0], x[1], x[2], True)
            ba.add_observation(str(shot_id), str(track_id),
                               *graph[shot_id][track_id]['feature'])

#    if config['bundle_use_gps']:

    # Always use GPS
    g = shot.metadata.gps_position
    ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                          shot.metadata.gps_dop)

    ba.set_loss_function(para['loss_function'],
                         para['loss_function_threshold'])
    ba.set_reprojection_error_sd(para['reprojection_error_sd'])
    ba.set_internal_parameters_prior_sd(
        para['exif_focal_sd'],
        para['principal_point_sd'],
        para['radial_distorsion_k1_sd'],
        para['radial_distorsion_k2_sd'],
        para['radial_distorsion_p1_sd'],
        para['radial_distorsion_p2_sd'],
        para['radial_distorsion_k3_sd'])
    ba.set_num_threads(para['processes'])

    ba.run()

    s = ba.get_shot(str(shot_id))
    shot.pose.rotation = [s.rx, s.ry, s.rz]
    shot.pose.translation = [s.tx, s.ty, s.tz]


def bundle_local(graph, reconstruction, gcp, central_shot_id, para):
    """Bundle adjust the local neighborhood of a shot."""

    interior, boundary = shot_neighborhood(
        graph, reconstruction, central_shot_id, para['local_bundle_radius'])

#    logger.debug(
#        'Local bundle sets: interior {}  boundary {}  other {}'.format(
#            len(interior), len(boundary),
#            len(reconstruction.shots) - len(interior) - len(boundary)))

    point_ids = set()
    for shot_id in interior:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    point_ids.add(track)

    ba = csfm.BundleAdjuster()

    for camera in reconstruction.cameras.values():
        _add_camera_to_bundle(ba, camera, constant=True)

    for shot_id in interior | boundary:
        shot = reconstruction.shots[shot_id]
        r = shot.pose.rotation
        t = shot.pose.translation
        ba.add_shot(
            str(shot.id), str(shot.camera.id),
            r[0], r[1], r[2],
            t[0], t[1], t[2],
            shot.id in boundary
        )

    for point_id in point_ids:
        point = reconstruction.points[point_id]
        x = point.coordinates
        ba.add_point(str(point.id), x[0], x[1], x[2], False)

    for shot_id in interior | boundary:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    ba.add_observation(str(shot_id), str(track),
                                       *graph[shot_id][track]['feature'])

    if para['bundle_use_gps']:
        for shot_id in interior:
            shot = reconstruction.shots[shot_id]
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

    if para['bundle_use_gcp'] and gcp:
        for observation in gcp:
            if observation.shot_id in interior:
                ba.add_ground_control_point_observation(
                    observation.shot_id,
                    observation.coordinates[0],
                    observation.coordinates[1],
                    observation.coordinates[2],
                    observation.shot_coordinates[0],
                    observation.shot_coordinates[1])

    ba.set_loss_function(para['loss_function'],
                         para['loss_function_threshold'])
    ba.set_reprojection_error_sd(para['reprojection_error_sd'])
    ba.set_internal_parameters_prior_sd(
        para['exif_focal_sd'],
        para['principal_point_sd'],
        para['radial_distorsion_k1_sd'],
        para['radial_distorsion_k2_sd'],
        para['radial_distorsion_p1_sd'],
        para['radial_distorsion_p2_sd'],
        para['radial_distorsion_k3_sd'])
    ba.set_num_threads(para['processes'])

    ba.run()

    for shot_id in interior:
        shot = reconstruction.shots[shot_id]
        s = ba.get_shot(str(shot.id))
        shot.pose.rotation = [s.rx, s.ry, s.rz]
        shot.pose.translation = [s.tx, s.ty, s.tz]

    for point in point_ids:
        point = reconstruction.points[point]
        p = ba.get_point(str(point.id))
        point.coordinates = [p.x, p.y, p.z]
        point.reprojection_error = p.reprojection_error

    report = {
        'brief_report': ba.brief_report(),
        'num_interior_images': len(interior),
        'num_boundary_images': len(boundary),
        'num_other_images': (len(reconstruction.shots)
                             - len(interior) - len(boundary)),
    }
    return report


def remove_outliers(graph, reconstruction, para):
    """Remove points with large reprojection error."""
    threshold = para['bundle_outlier_threshold']
    if threshold > 0:
        outliers = []
        for track in reconstruction.points:
            error = reconstruction.points[track].reprojection_error
            if error > threshold:
                outliers.append(track)
        for track in outliers:
            del reconstruction.points[track]
#        logger.info("Removed outliers: {}".format(len(outliers)))


def retriangulate(graph, reconstruction, para):
    """Retrianguate all points"""
    report = {}
    report['num_points_before'] = len(reconstruction.points)
    threshold = para['triangulation_threshold']
    min_ray_angle = para['triangulation_min_ray_angle']
    triangulator = TrackTriangulator(graph, reconstruction)
    tracks, images = tracks_and_images(graph)
    for track in tracks:
        triangulator.triangulate(track, threshold, min_ray_angle)
    report['num_points_after'] = len(reconstruction.points)
    return report


def grow_reconstruction(para, graph, reconstruction, images, gcp, exif, ref):
    """Incrementally add shots to an initial reconstruction."""
#    print('bundling...')
#    bundle(graph, reconstruction, None, para)
#    print('align reconstruction...')
#    align_reconstruction(reconstruction, gcp, para)
#
#    should_bundle = ShouldBundle(para, reconstruction)
#    print('should bundling...')
#    should_retriangulate = ShouldRetriangulate(para, reconstruction)
    report = {
        'steps': [],
    }
    while True:
        if para['save_partial_reconstructions']:
            paint_reconstruction(para, graph, reconstruction)
            dataset.save_reconstruction(
                [reconstruction], 'reconstruction.{}.json'.format(
                    time.time().isoformat().replace(':', '_')))

        common_tracks = reconstructed_points_for_images(graph, reconstruction,
                                                        images)
        if not common_tracks:
            break

#        logger.info("-------------------------------------------------------")
        for image, num_tracks in common_tracks:
            ok, resrep = resect(exif, graph, reconstruction, image, ref, para)
            if ok:
#                logger.info("Adding {0} to the reconstruction".format(image))
                step = {
                    'image': image,
                    'resection': resrep,
                    'memory_usage': context.current_memory_usage()
                }
                report['steps'].append(step)
                images.remove(image)

                np_before = len(reconstruction.points)
                triangulate_shot_features(
                    graph, reconstruction, image,
                    para['triangulation_threshold'],
                    para['triangulation_min_ray_angle'])

                np_after = len(reconstruction.points)
                step['triangulated_points'] = np_after - np_before

#                if should_bundle.should(reconstruction):
#                    brep = bundle(graph, reconstruction, None, para)
#                    step['bundle'] = brep
#                    remove_outliers(graph, reconstruction, para)
#                    align_reconstruction(reconstruction, gcp,
#                                         para)
#                    should_bundle.done(reconstruction)
#                else:
#                    if para['local_bundle_radius'] > 0:
#                        brep = bundle_local(graph, reconstruction, None, image,
#                                            para)
#                        step['local_bundle'] = brep
#
#                if should_retriangulate.should(reconstruction):
#                    rrep = retriangulate(graph, reconstruction, para)
#                    step['retriangulation'] = rrep
#                    bundle(graph, reconstruction, None, para)
#                    should_retriangulate.done(reconstruction)
                paint_reconstruction(para, graph, reconstruction)
                break
        else:
            print('Some images can not be added')
            break

    print('bundling...')
    bundle(graph, reconstruction, gcp, para)
    print('aligning...')
    align_reconstruction(reconstruction, gcp, para)
    paint_reconstruction(para, graph, reconstruction)
    return reconstruction, report


def bootstrap_reconstruction(camera_model, exif, ref, graph, im1, im2, p1, p2, para):
    """Start a reconstruction using two shots."""
    report = {
        'image_pair': (im1, im2),
        'common_tracks': len(p1),
    }

    cameras = camera_model
    # Taking camera model of image1 and image2 . Here this allows us...
    # ... to take different camera models in single project

    camera1 = cameras[exif[im1]['camera']]
    camera2 = cameras[exif[im2]['camera']]

    threshold = para['five_point_algo_threshold']
    min_inliers = para['five_point_algo_min_inliers']
    R, t, inliers, report['two_view_reconstruction'] = \
        two_view_reconstruction_general(p1, p2, camera1, camera2, threshold)

    if len(inliers) <= 5:
        report['decision'] = "Could not find initial motion"
        return None, report

    reconstruction = types.Reconstruction()
    reconstruction.cameras = cameras

    shot1 = types.Shot()
    shot1.id = im1
    shot1.camera = camera1
    shot1.pose = types.Pose()
    shot1.metadata = get_image_metadata(exif, ref, im1)
    reconstruction.add_shot(shot1)

    shot2 = types.Shot()
    shot2.id = im2
    shot2.camera = camera2
    shot2.pose = types.Pose(R, t)
    shot2.metadata = get_image_metadata(exif, ref, im2)
    reconstruction.add_shot(shot2)

    triangulate_shot_features(
        graph, reconstruction, im1,
        para['triangulation_threshold'],
        para['triangulation_min_ray_angle'])
    
    report['triangulated_points'] = len(reconstruction.points)

    if len(reconstruction.points) < min_inliers:
        report['decision'] = "Initial motion did not generate enough points"
        return None, report

    bundle_single_view(graph, reconstruction, im2, para)
    retriangulate(graph, reconstruction, para)
    bundle_single_view(graph, reconstruction, im2, para)

    report['decision'] = 'Success'
    report['memory_usage'] = context.current_memory_usage()
    return reconstruction, report


def two_view_reconstruction_general(p1, p2, camera1, camera2, threshold):
    """Reconstruct two views from point correspondences.

    These will try different reconstruction methods and return the
    results of the one with most inliers.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    R_5p, t_5p, inliers_5p = two_view_reconstruction(
        p1, p2, camera1, camera2, threshold)

    report = {
        '5_point_inliers': len(inliers_5p),
    }

    report['method'] = '5_point'
    return R_5p, t_5p, inliers_5p, report


def two_view_reconstruction(p1, p2, camera1, camera2, threshold):
    """Reconstruct two views using the 5-point method.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    # Note on threshold:
    # See opengv doc on thresholds here:
    #   http://laurentkneip.github.io/opengv/page_how_to_use.html
    # Here we arbitrarily assume that the threshold is given for a camera of
    # focal length 1.  Also, arctan(threshold) \approx threshold since
    # threshold is small
    T = run_relative_pose_ransac(
        b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000)
    R = T[:, :3]
    t = T[:, 3]
    inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    T = run_relative_pose_optimize_nonlinear(b1[inliers], b2[inliers], t, R)
    R = T[:, :3]
    t = T[:, 3]
    inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), -R.T.dot(t), inliers


def run_relative_pose_ransac(b1, b2, method, threshold, iterations):
    return pyopengv.relative_pose_ransac(b1, b2, method, threshold, iterations)


def _two_view_reconstruction_inliers(b1, b2, R, t, threshold):
    """Compute number of points that can be triangulated.

    Args:
        b1, b2: Bearings in the two images.
        R, t: Rotation and translation from the second image to the first.
              That is the opengv's convention and the opposite of many
              functions in this module.
        threshold: max reprojection error in radians.
    Returns:
        array: Inlier indices.
    """
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()  # Relative to camera 1
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]  # Normalized point

    br2 = R.T.dot((p - t).T).T  # Relative to camera 2
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]  # Normalized point

    # Checking if reprojected error is less than threshold in camera1
    ok1 = np.linalg.norm(br1 - b1, axis=1) < threshold
    # Checking if reprojected error is less than threshold in camera2
    ok2 = np.linalg.norm(br2 - b2, axis=1) < threshold
    return np.nonzero(ok1 * ok2)[0]


def run_relative_pose_optimize_nonlinear(b1, b2, t, R):
    return pyopengv.relative_pose_optimize_nonlinear(b1, b2, t, R)


def align_reconstruction(reconstruction, gcp, config):
    """Align a reconstruction with GPS and GCP data."""
    res = align_reconstruction_similarity(reconstruction, gcp, config)
    if res:
        s, A, b = res
        apply_similarity(reconstruction, s, A, b)


def apply_similarity(reconstruction, s, A, b):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    :param reconstruction: The reconstruction to transform.
    :param s: The scale (a scalar)
    :param A: The rotation matrix (3x3)
    :param b: The translation vector (3)
    """
    # Align points.
    for point in reconstruction.points.values():
        Xp = s * A.dot(point.coordinates) + b
        point.coordinates = Xp.tolist()

    # Align cameras.
    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        t = np.array(shot.pose.translation)
        Rp = R.dot(A.T)
        tp = -Rp.dot(b) + s * t
        shot.pose.set_rotation_matrix(Rp)
        shot.pose.translation = list(tp)


def align_reconstruction_similarity(reconstruction, gcp, config):
    """Align reconstruction with GPS and GCP data.

    Config parameter `align_method` can be used to choose the alignment method.
    Accepted values are
     - navie: does a direct 3D-3D fit
     - orientation_prior: assumes a particular camera orientation
    """
    align_method = config['align_method']
    if align_method == 'orientation_prior':
        return align_reconstruction_orientation_prior_similarity(
            reconstruction, config)
    elif align_method == 'naive':
        return align_reconstruction_naive_similarity(reconstruction, gcp)


def align_reconstruction_naive_similarity(reconstruction, gcp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    X, Xp = [], []

    # Get Ground Control Point correspondences
    if gcp:
        triangulated, measured = triangulate_all_gcp(reconstruction, gcp)
        X.extend(triangulated)
        Xp.extend(measured)

    # Get camera center correspondences
    for shot in reconstruction.shots.values():
        X.append(shot.pose.get_origin())
        Xp.append(shot.metadata.gps_position)

    if len(X) < 3:
        return

    # Compute similarity Xp = s A X + b
    X = np.array(X)
    Xp = np.array(Xp)
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)

    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b


def align_reconstruction_orientation_prior_similarity(reconstruction, config):
    """Align with GPS data assuming particular a camera orientation.

    In some cases, using 3D-3D matches directly fails to find proper
    orientation of the world.  That happends mainly when all cameras lie
    close to a straigh line.

    In such cases, we can impose a particular orientation of the cameras
    to improve the orientation of the alignment.  The config parameter
    `align_orientation_prior` can be used to specify such orientation.
    Accepted values are:
     - no_roll: assumes horizon is horizontal on the images
     - horizontal: assumes cameras are looking towards the horizon
     - vertical: assumes cameras are looking down towards the ground
    """
    X, Xp = [], []
    orientation_type = config['align_orientation_prior']
    onplane, verticals = [], []
    for shot in reconstruction.shots.values():
        X.append(shot.pose.get_origin())
        Xp.append(shot.metadata.gps_position)
        R = shot.pose.get_rotation_matrix()
        x, y, z = get_horizontal_and_vertical_directions(
            R, shot.metadata.orientation)
        if orientation_type == 'no_roll':
            onplane.append(x)
            verticals.append(-y)
        elif orientation_type == 'horizontal':
            onplane.append(x)
            onplane.append(z)
            verticals.append(-y)
        elif orientation_type == 'vertical':
            onplane.append(x)
            onplane.append(y)
            verticals.append(-z)

    X = np.array(X)
    Xp = np.array(Xp)

    # Estimate ground plane.
    p = multiview.fit_plane(X - X.mean(axis=0), onplane, verticals)
    Rplane = multiview.plane_horizontalling_rotation(p)
    X = Rplane.dot(X.T).T

    # Estimate 2d similarity to align to GPS
    if (len(X) < 2 or
            X.std(axis=0).max() < 1e-8 or     # All points are the same.
            Xp.std(axis=0).max() < 0.01):      # All GPS points are the same.
        # Set the arbitrary scale proportional to the number of cameras.
        s = len(X) / max(1e-8, X.std(axis=0).max())
        A = Rplane
        b = Xp.mean(axis=0) - X.mean(axis=0)
    else:
        T = tf.affine_matrix_from_points(X.T[:2], Xp.T[:2], shear=False)
        s = np.linalg.det(T[:2, :2])**0.5
        A = np.eye(3)
        A[:2, :2] = T[:2, :2] / s
        A = A.dot(Rplane)
        b = np.array([
            T[0, 2],
            T[1, 2],
            Xp[:, 2].mean() - s * X[:, 2].mean()  # vertical alignment
        ])
    return s, A, b


def get_horizontal_and_vertical_directions(R, orientation):
    """Get orientation vectors from camera rotation matrix and orientation tag.

    Return a 3D vectors pointing to the positive XYZ directions of the image.
    X points to the right, Y to the bottom, Z to the front.
    """
    # See http://sylvana.net/jpegcrop/exif_orientation.html
    if orientation == 1:
        return R[0, :], R[1, :], R[2, :]
    if orientation == 2:
        return -R[0, :], R[1, :], -R[2, :]
    if orientation == 3:
        return -R[0, :], -R[1, :], R[2, :]
    if orientation == 4:
        return R[0, :], -R[1, :], R[2, :]
    if orientation == 5:
        return R[1, :], R[0, :], -R[2, :]
    if orientation == 6:
        return -R[1, :], R[0, :], R[2, :]
    if orientation == 7:
        return -R[1, :], -R[0, :], -R[2, :]
    if orientation == 8:
        return R[1, :], -R[0, :], R[2, :]
    return R[0, :], R[1, :], R[2, :]


def triangulate_single_gcp(reconstruction, observations):
    """Triangulate one Ground Control Point."""
    reproj_threshold = 0.004
    min_ray_angle_degrees = 2.0

    os, bs = [], []
    for o in observations:
        if o.shot_id in reconstruction.shots:
            shot = reconstruction.shots[o.shot_id]
            os.append(shot.pose.get_origin())
            b = shot.camera.pixel_bearing(np.asarray(o.shot_coordinates))
            r = shot.pose.get_rotation_matrix().T
            bs.append(r.dot(b))

    if len(os) >= 2:
        e, X = csfm.triangulate_bearings_midpoint(
            os, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
        return X


def triangulate_all_gcp(reconstruction, gcp_observations):
    """Group and triangulate Ground Control Points seen in 2+ images."""
    groups = defaultdict(list)
    for o in gcp_observations:
        groups[tuple(o.lla)].append(o)

    triangulated, measured = [], []
    for observations in groups.values():
        x = triangulate_single_gcp(reconstruction, observations)
        if x is not None:
            triangulated.append(x)
            measured.append(observations[0].coordinates)

    return triangulated, measured
    
def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    
def project(points, camera_params, k, width, height):
    """Convert 3-D points to 2-D by projecting onto images.
    x = RX + t, where x is 3d point in camera frame, R is rotational matrix,
    and X is 3D coordinate, t is translation vector"""
    
    # RX
    points_proj = rotate(points, camera_params[:, :3])
    # RX + t    
    points_proj += camera_params[:, 3:6]
    
    points_proj = np.matmul(k, points_proj.T)
    points_proj = points_proj.T

    # X/X[2]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    return points_proj

def fun(x0, n_cameras, n_points, camera_index, point_index, point_2d, camera_model):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = x0[:n_cameras * 14].reshape((n_cameras, 14))
    point_3d = x0[n_cameras * 14:].reshape((n_points, 3))
    
    camera_name = camera_model.keys()
    cam = camera_model[camera_name[0]]
    width = cam.width
    height = cam.height
    k = cam.get_K_in_pixel_coordinates()
    
    point_2d = denormalized_image_coordinates(point_2d,width,height)
    points_proj = project(point_3d[point_index], camera_params[camera_index], k, width, height)
    return (points_proj - point_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 14 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(14):
        A[2 * i, camera_indices * 14 + s] = 1
        A[2 * i + 1, camera_indices * 14 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 14 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 14 + point_indices * 3 + s] = 1

    return A

def mapcam(reconstruct):
    index = 0
    mapshot = {}
    for i in reconstruct:
        shot = i['shots']
        for j in shot:
            mapshot[j] = index
            index  = index +  1
    return mapshot

def cam2index(cam, mapshot):       
    return mapshot[cam]

def index2cam(index,mapshot):
    cam = []    
    for _cam, _ind in mapshot.iteritems():
        if _ind == index:
            cam.append(_cam)
    return cam
    
def cam_param(cam_ind, reconstruct):
    
    """ focal_pixel = (focal_mm / sensor_width_mm) * image_width_in_pixels"""
    cam_param_ = np.zeros([len(np.unique(cam_ind)),14])
    mapshot = mapcam(reconstruct)
    
    for rec in reconstruct:
        param = rec['cameras']
        camera_model = param.keys()
        k1 = float(param[camera_model[0]]['k1'])
        k2 = float(param[camera_model[0]]['k2'])
        k3 = float(param[camera_model[0]]['k3'])
        p1 = float(param[camera_model[0]]['p1'])
        p2 = float(param[camera_model[0]]['p2'])
        width = float(param[camera_model[0]]['width'])
        f = float(param[camera_model[0]]['focal_x']) * width
        cx = float(param[camera_model[0]]['c_x'])
        cy = float(param[camera_model[0]]['c_y'])
        for cam_ in rec['shots']:
            index = cam2index(cam_,mapshot)
            r = np.asarray(rec['shots'][cam_]['rotation'])
            t = np.asarray(rec['shots'][cam_]['translation'])
#            c = np.asarray(rec['shots'][cam_]['gps_position'])
            cam_param_[index,0:3] = r
            cam_param_[index,3:6] = t
            cam_param_[index,6] = f
            cam_param_[index,7] = cx
            cam_param_[index,8] = cy
            cam_param_[index,9] = k1
            cam_param_[index,10] = k2
            cam_param_[index,11] = k3
            cam_param_[index,12] = p1
            cam_param_[index,13] = p2 
    return cam_param_
  
def arr_param(reconstruct, track_graph, remaining_images):
    pt_3d = []
    pt_color = []
    pt_2d  = []
    cam_ind = []
    pt_ind = []
    ind = -1
    mapshot = mapcam(reconstruct)

    for rec in reconstruct:
        pt = rec['points']
        # Mapping shots(cameras)
        for track_ in pt:
            pt_3d.append(pt[track_]['coordinates'])
            pt_color.append(pt[track_]['color'])
            tr = track_graph[track_]
            ind = ind + 1
            for im in tr:
                if len(set.intersection(set([im]),remaining_images)) != 0:
                    print('Skipping remaining images %s'%(im))
                    continue
                px, py = tr[im]['feature']
                pt_2d.append([px,py])
                pt_ind.append(ind)
                cam_index = cam2index(im,mapshot)
                cam_ind.append(cam_index)
            
    return np.asarray(pt_3d), np.asarray(pt_color), \
           np.asarray(pt_2d), np.asarray(cam_ind), np.asarray(pt_ind)
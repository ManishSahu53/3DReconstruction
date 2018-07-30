""" Incremental bundle adjustment """
import pyopengv
import cv2
import numpy as np
from src import context, types, dataset
from six import iteritems

def compute_image_pair(track_dict, list_ ,processes): # list = [exif,camera_model]
    """All matched image pairs sorted by reconstructability."""
    args = _pair_reconstructability_arguments(track_dict, list_) # list = [exif,camera_model]
    result = context.parallel_map(_compute_pair_reconstructability, args, processes)
    result = list(result)
    pairs = [(im1, im2) for im1, im2, r in result if r > 0]
    score = [r for im1, im2, r in result if r > 0]
    order = np.argsort(-np.array(score))
    return [pairs[o] for o in order]
    
def _pair_reconstructability_arguments(track_dict, list): # list = [exif,camera_model]
    threshold = 4 * 0.004 # Outlier threshold (in pixels) for essential matrix estimation #data.config['five_point_algo_threshold']
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

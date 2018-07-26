""" Creating tracks from matches"""

from unionfind import UnionFind
import numpy as np

import networkx as nx
from collections import defaultdict
from itertools import combinations
from six import iteritems

    
def _good_track(track, min_length):
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True


def create_tracks_graph(feature, colors, match):
    """Link matches into tracks."""
    uf = UnionFind()
    for im1, im2 in match:
        for f1, f2 in match[im1, im2]:
            uf.union((im1, f1), (im2, f2))

    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    tracks = [t for t in sets.values() if _good_track(t, 5)] #5 is minimum track length

    tracks_graph = nx.Graph()
    for track_id, track in enumerate(tracks):
        for image, featureid in track:
#            if image not in feature:
#                continue
            x, y = feature[image][featureid]
            r, g, b = colors[image][featureid]
            tracks_graph.add_node(image, bipartite=0)
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(image,
                                  str(track_id),
                                  feature=(x, y),
                                  feature_id=featureid,
                                  feature_color=(float(r), float(g), float(b)))

    return tracks_graph


def tracks_and_images(graph):
    """List of tracks and images in the graph."""
    tracks, images = [], []
    for n in graph.nodes(data=True):
        if n[1]['bipartite'] == 0:
            images.append(n[0])
        else:
            tracks.append(n[0])
    return tracks, images


def common_tracks(graph, im1, im2):
    """List of tracks observed in both images.

    Args:
        graph: tracks graph
        im1: name of the first image
        im2: name of the second image

    Returns:
        tuple: tracks, feature from first image, feature from second image
    """
    t1, t2 = graph[im1], graph[im2]
    tracks, p1, p2 = [], [], []
    for track in t1:
        if track in t2:
            p1.append(t1[track]['feature'])
            p2.append(t2[track]['feature'])
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


def all_common_tracks(graph, tracks, include_features=True, min_common=50):
    """List of tracks observed by each image pair.

    Args:
        graph: tracks graph
        tracks: list of track identifiers
        include_features: whether to include the features from the images
        min_common: the minimum number of tracks the two images need to have
            in common

    Returns:
        tuple: im1, im2 -> tuple: tracks, features from first image, features
        from second image
    """
    track_dict = defaultdict(list)
    for track in tracks:
        track_images = sorted(graph[track].keys())
        for im1, im2 in combinations(track_images, 2):
            track_dict[im1, im2].append(track)

    common_tracks = {}
    for k, v in iteritems(track_dict):
        if len(v) < min_common:
            continue
        im1, im2 = k
        if include_features:
            p1 = np.array([graph[im1][tr]['feature'] for tr in v])
            p2 = np.array([graph[im2][tr]['feature'] for tr in v])
            common_tracks[im1, im2] = (v, p1, p2)
        else:
            common_tracks[im1, im2] = v
    return common_tracks
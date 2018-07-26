# Tracks documentation

## Introduction
This command links the matches between pairs of images to build feature point tracks. A track is a set of feature points from different images that represent to the same pysical point.  A track represents a physical point which is seen in multiple images. For example - If a point (P1) is a track and it is detected in 5 images (using features extraction method), then this track is a list of length 5. Each contains name of image and its feature id representing point P1. If we want to know location of P1 in image then simply look in keypoin object (`kp[feature_id].pt`). It will give *X* and *Y* coordinate of P1 in that image.

There are two biparite (two sets) in networkx. 
1. Images
2. Track_ID

This variable is a dictionary which has key of *Images* and value as *Track_ID*

## Output
The tracks are stored in the tracks.csv file. It has 8 columns.
> ImageName, Track_ID, Feature_ID, X, Y, Red, Green, Blue

* **ImageName** - Gives name of image dealing with
* **Track_ID** - When converting feature points to tracks, a large number of points get converted. So each feature which is converted into track has an ID (Track_ID). Each ID represents a separate track. Each track contains information given in first paragraph.
* **Feature_ID** - This ID came from feature detection step. For example if ORB feature is used for extraction and it has extracted 40,000 features then there will be 40,000 *Feature_IDs*. Each ID represents a keypoint.
* **X, Y** - Represents coordinate is pixel coordinates.
* **R,G,B** - It give color information for that feature.

## Internal/Dependent Functions
* **match**  - This match is matches of features. It is also a dictionary having two keys. *Master Image* and *Pair Image* and this key contains value of their feature matches. It has two columns, **first is of feature Id of Master Image and second is feature Id of Pair Image.**. 
* **feature** - This is dictionary of features where Key represents Image Name and vlaue represents array of features.
* **UnionFind** - This is same as union is set theory of mathematics. This will take of all the features matched. This variable contains all the matches column wise in single variable *uf*. For example - 

```
for i in uf:
	p = uf\[i]\
```
i = ('DJI_0088.JPG', 3435)

p = ('DJI_0089.JPG', 1483)

This means 88.JPG's feature_id number 2425 is matched with 99.JPG's feature_id number 1483.

* **sets** -  set is also a dictionary where key is p and value is i. 

```
// If p feature already exist then append else create a dict with key is p and value is i. This will append commnon points

if p in sets: 
	sets(p).append(i)
else:
	sets(p) = (i)
```

This will append all the same features. Length of *sets\[p]\ will give us feature is seen in that many images.

* **tracks** - In the SFM step we want to take features which are visible in >3 images so that good triagulation is possible. So we need to filer out tracks whose length is greater than 3 . This 3 is a parameter, here it is taken to 4.

* **tracks_graph** - A networkx graph is created for the tracks. There are 2 sets of nodes (Bipartite). 

* Images
* Track_ID

Node set 1 will match(edge connections) with Node set 2. For example - A image has number of features, and thus matches. These matches are converted into tracks. So each image has number of tracks (ID). So the graph represents this image and track_ID relationship. 

""" Extracting Exif information from the images"""
import os,json,utm,math

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
    
# For converting projected to geographic coordinate system
def xy2latlong(easting,northing,zone,hemi):
    lat,long = utm.to_latlon( easting,northing,zone,hemi);
    return lat,long

# For converting geographic to projected coordinate system
def latlong2utm(lat,long):
        coord = utm.from_latlon(lat, long);
        easting =  coord[0];
        northing = coord[1];
        zone = coord[2];
        hemi = coord[3];
        return easting,northing,zone,hemi
        
def distance(lat1,long1,lat2,long2):
    easting1,northing1,zone1,hemi1 = latlong2utm(lat1,long1)
    easting2,northing2,zone2,hemi2 = latlong2utm(lat2,long2)
    return math.sqrt((easting1-easting2)**2 + (northing1-northing2)**2)

def get_neighbour(coord,list_coord,num_neighbour):
    # coord = [ lat[0],long[0],image[0]]
    # list_coord = [lat,long,image]
    neighbour_dist =[];
    sorted_dist = [];
    neighbour_image =[];
    
    for i in range(len(list_coord[0])):
        dist = distance(coord[0],coord[1],list_coord[0][i],list_coord[1][i])
        neighbour_dist.append(dist)
        sorted_dist.append(dist)
    sorted_dist.sort(key=float)
    
    # First element will always be the image itself as distance will be zero
    # with respect to self. So adding [neighbour+1] to get the 
    for j in range(num_neighbour+1):
        neighbour_image.append(list_coord[2][neighbour_dist.index(sorted_dist[j])]);
    return neighbour_image

""" Matching features from images, feature_matching.py"""

import logging
reload(logging)
  
def checkdir(path):
    if path[-1] != '/':
        path = path  + '/'        
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")
    
def tojson(dictA,file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f,indent=4, separators=(',', ': '), ensure_ascii=False,encoding='utf-8')
        
def pickle_keypoints(keypoints, descriptors,color):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        i = i +1
        temp_array.append(temp)
    return [temp_array,[color]]

def unpickle_keypoints(image,path_feature):
    path_feature = os.path.realpath(path_feature);
    
    image = os.path.splitext(os.path.basename(image))[0];
    image = os.path.join(path_feature,image+'.npy');
    array = np.load( open( image, "rb" ) );

    keypoints = []
    descriptors = []
    color = array[1]
    array = array[0]
    
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors),color

def num2para(number):
    if number == 1:
        parameter_ = 'ANN/Hamming'
    return parameter_

def num2method(number):
    if number == 1:
        method_ = 'sift'
    if number == 2:
        method_ = 'surf'
    if number == 3:
        method_ = 'orb'
    if number == 4:
        method_ = 'brisk'
    if number == 5:
        method_ = 'akaze'
    if number == 6:
        method_ = 'StarBrief'
    return method_

def numfeature(path):
    list_feature = []
#    Searching of features
    for root,dirs,files in os.walk(path):
        if len(files) == 0:
            sys.exit('No features found in "%s" directory'%(path))
            break;
        for file_name in files:
            if file_name.endswith(('.npy','.NPY','Npy')):
                list_feature.append((os.path.join(path,file_name)));
    return len(list_feature)

def _convert_match_to_vector(match):
    """Convert Dmatch object to matrix form."""
    match_vector = np.zeros((len(match), 2), dtype=np.int)
    k = 0
    for mm in match:
        # mm.query will give index of mm match in keypoint detector.
        # Example- kp1[mm.queryIdx].pt will give,
        # x and y coordinate of image corresponding keypoint1
    
        match_vector[k, 0] = mm.queryIdx  # Train is master image                                         
        match_vector[k, 1] = mm.trainIdx  # Query is pair image
        k = k+1
    return match,match_vector

def robust_match_ratio_test(kp1,kp2,match,ratio):
    valid_match = []
    pts1 = []
    pts2 = []
    """Rejects matches based on popular lowe ratio test"""
    for _k,(m,n) in enumerate(match):
        if m.distance < ratio*n.distance:
            valid_match.append(m)
            pts1.append(kp1[m.queryIdx].pt) # Query is master image, Not sure
            pts2.append(kp2[m.trainIdx].pt) # Train is pair  image, Not sure
    return valid_match,np.asarray(pts1,dtype = np.int32),np.asarray(pts2,dtype = np.int32)
    
def robust_match_fundamental(pts1, pts2, valid_match):
    """Filter matches by estimating the Fundamental matrix via RANSAC."""
    if len(valid_match) < 8:
        return np.array([])
    FM_RANSAC = cv2.FM_RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, FM_RANSAC)
    index = mask.ravel() ==1
    if F[2, 2] == 0.0:
        return []
    return list(compress(valid_match, index)),pts1[index],pts2[index]   

def _convert_matches_to_vector(match):
    """Convert Dmatch object to matrix form."""
    match_vector = np.zeros((len(match), 2), dtype=np.int)
    k = 0
    for mm in match:
        match_vector[k, 0] = mm.queryIdx
        match_vector[k, 1] = mm.trainIdx
        k = k+1
    return match_vector

def match_feature(imagepair):#,path_feature,path_output):

    try:
        match_data = []
#        for i in range(_num_image):
        _pair_logging = [];
        _start_time = time.time()
        master_im = imagepair[0];
        _match = {};
        for j in range(1,_num_pair):
            pair_im = imagepair[j];
            valid_match = []
            
#                Retrieve Keypoint Features
            kp1, des1, color = unpickle_keypoints(master_im,path_feature)
            kp2, des2, color = unpickle_keypoints(pair_im,path_feature)
            
#            Changing matching method for SIFT/SURF and Binary features.
#            Binary doesn't have ANN option and so Brute Force method is used.
            
            if args.feature == 1 or args.feature == 2:
#                FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50)   # or pass empty dictionary
            
#                loading saved keypoints and descriptors
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                match_ = flann.knnMatch(des1,des2,k=2)
                
#                store all the good matches as per Lowe's ratio test.
                valid_match,pts1,pts2 = robust_match_ratio_test(kp1,kp2,match_,ratio)
                
#                Robut fundamental outlier removal
                valid_match,pts1,pts2 = robust_match_fundamental(pts1,pts2,valid_match)
                match_index = _convert_matches_to_vector(valid_match)

                
            elif args.feature == 3 or args.feature == 4 or args.feature == 5 or args.feature == 6: # For binary descriptors, using hamming distance
                
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

                # Match descriptors.
                match_ = bf.knnMatch(des1,des2,k=2)
                
#                store all the good matches as per Lowe's ratio test.
                valid_match,pts1,pts2 = robust_match_ratio_test(kp1,kp2,match_,ratio)
                
#                Robut fundamental outlier removal
                valid_match,pts1,pts2 = robust_match_fundamental(pts1,pts2,valid_match)
                match_index = np.array(_convert_matches_to_vector(valid_match),dtype = int)

#                Printing and logging info
#            logger.info('Image %s processed, took : %s sec per thread'
#            %(os.path.basename(image),str(_time_taken)))
            
            _pair_logging.append({"Pair im" : pair_im,
                                 "Total matches" : len(match_),
                                 "Valid matches" : len(valid_match),
                                 "Inlier percentage" : round(float(len(valid_match))*100,2)/len(match_)})
            
#           Appending matches data for saving 
            match_data.append([[valid_match,match_index],[pts1,pts2],[master_im,pair_im]])
            _match[pair_im] = match_index
            
#         Updating path_data to path_feature
        print(master_im)
        path_match = os.path.join(path_data,method_feature);
        checkdir(path_match)
        
#         Saving features as numpy compressed array
        path_match = os.path.join(path_match, os.path.splitext(os.path.basename(master_im))[0])
#        np.save(path_match,match_data)_match
        np.save(path_match,_match)
        _end_time = time.time()
        _time_taken = round(_end_time - _start_time,1) 
        

        _master_logging = {"Master Image" : master_im,
                "Pair Images" : _pair_logging,
                "Features extraction method" : method_feature,
                "Threshold Ration" : ratio,
                "Similarity matric" : num2para(parameter),
                "Time" : _time_taken};
                
#         Saving corresponding matches of master image to file 
        tojson(_master_logging,os.path.join(path_logging,os.path.splitext(os.path.basename(master_im))[0] + '.json'))
   
        match_report.append(_master_logging)
           
    except KeyboardInterrupt:
#        logger.fatal('KeyboardInterruption, Terminated')
        sys.exit('KeyboardInterruption, Terminated')
        pause()
  
    return match_report

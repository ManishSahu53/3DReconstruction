""" Extracting features from images, extract_feature.py"""
import logging
reload(logging)
  
def checkdir(path):
    if path[-1] != '/':
        path = path  + '/'        
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        
def pickle_keypoints(keypoints, descriptors,color):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        i = i +1
        temp_array.append(temp)
    return [temp_array,[color]]

def unpickle_keypoints(_array):
    keypoints = []
    descriptors = []
    color = _array[1]
    array = _array[0]
    
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors),color

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
    
    
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")
    
def tojson(dictA,file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f,indent=4, separators=(',', ': '), ensure_ascii=False,encoding='utf-8')
        
def create_logger(name):
    """Create a logger

    args: name (str): name of logger

    returns: logger (obj): logging.Logger instance
    """
    fmt = logging.Formatter('%(asctime)s - %(name)s -'
                            ' %(levelname)s -%(message)s')
    hdl = logging.FileHandler(name+'.log')

    return logging
    
def kp2xy(kp):
    point = np.array([(i.pt[0], i.pt[1]) for i in kp])
    x = point[:,0].round().astype(int)
    y = point[:,1].round().astype(int)
    return x,y

    
          
def extract_feature(image):
    try:
#        for image in list_image:
        im = cv2.imread(image);
        _gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        _start_time = time.time();
        
        if method == 1: # Sift 
            method_ = 'sift/'
            if num_feature == 0:
                sift_ = cv2.xfeatures2d_SIFT.create();
            else:
                sift_ = cv2.xfeatures2d_SIFT.create(nfeatures = num_feature); # Predefining number of features
                
            kp, des = sift_.detectAndCompute(_gray,None);
            x,y = kp2xy(kp)
            color = im[y,x] # Since im[row,column]
            
        if method ==2: # Surf
            method_ = 'surf/'
            if num_feature == 0:
                surf_ = cv2.xfeatures2d.SURF_create()
            else:
                surf_ = cv2.xfeatures2d.SURF_create(2000)# ~ 50000 features 
                
            kp, des = surf_.detectAndCompute(_gray,None);
            x,y = kp2xy(kp)
            color = im[y,x] # Since im[row,column]
             
        if method ==3: # ORB
            method_ = 'orb/'
            if num_feature ==0:
                 orb_ = cv2.ORB_create()
            else:  
                 orb_ = cv2.ORB_create(nfeatures = num_feature) # Predefining number of features
                 
            kp,des = orb_.detectAndCompute(_gray,None);
            x,y = kp2xy(kp)
            color = im[y,x] # Since im[row,column]
            
        if method == 4: # Brisk
            method_ = 'brisk/' 
            if num_feature ==0:
                  brisk_ = cv2.BRISK_create()
            else:
                brisk_ = cv2.BRISK_create(75) # ~ 50,000 features 
                
            kp, des = brisk_.detectAndCompute(_gray,None);
            x,y = kp2xy(kp)
            color = im[y,x] # Since im[row,column]
            
        if method == 5: # AKAZE
            method_ = 'akaze/'
            akaze_ = cv2.AKAZE_create();
            kp, des = akaze_.detectAndCompute(_gray,None);
            
            x,y = kp2xy(kp)
            color = im[y,x] # Since im[row,column]
            
        if method ==6: # Star + BRIEF
            method_ = 'StarBrief/'
            star_ = cv2.xfeatures2d.StarDetector_create() # ~ 80,000 features
            brief_ = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp = star_.detect(_gray,None);
            kp, des = brief_.compute(_gray, kp)
           
            x,y = kp2xy(kp)
            color = im[y,x] # Since im[row,column]
            
#           Store and Retrieve keypoint features
        temp = pickle_keypoints(kp, des, color)
        
#        Updating output path
        path_feature = os.path.join(path_data,method_);
        checkdir(path_feature)
        
#           Saving features as numpy compressed array
        path_feature = path_feature + os.path.splitext(os.path.basename(image))[0]
        np.save(path_feature,temp)
        
#            Printing and logging info
        _end_time = time.time()
        _time_taken = round(_end_time - _start_time,1)
        
#        logger.info('Image %s processed, took : %s sec per thread'
#        %(os.path.basename(image),str(_time_taken)))
                
#        Saving number of feature dectected to disk 
        feature = {"Number of Features" : len(kp),
                "Image" : os.path.splitext(os.path.basename(image))[0],
                "Time" : _time_taken,
                "Method" : method_}
                
        tojson(feature,os.path.join(path_logging,os.path.splitext(os.path.basename(image))[0] + '.json'))
        
        print('finished processing %s and took %s sec per thread'
        %(os.path.splitext(os.path.basename(image))[0],_time_taken) )
        
#        Counting Features
        count_feature.append(len(kp));
        append_image.append(os.path.splitext(os.path.basename(image))[0]); 
        append_time.append(_time_taken)
        append_method.append(method_)
        
#        Reporting
        features = {"Number of Features": count_feature,
                    "Image": append_image,
                    "Time": append_time,
                    "Method": append_method};
         
    except KeyboardInterrupt:
#        logger.fatal('KeyboardInterruption, Terminated')
        sys.exit('KeyboardInterruption, Terminated')
        pause()
  
#       Returns list of images
    return features


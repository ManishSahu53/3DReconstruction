def list_image(path,path_logging):
    import time
    import logging
    import os
    # Starting recording time
    start_time = time.time();
    
    list_image = []; # Getting list of images
    # Logging starts
    logging.basicConfig(format='%(asctime)s %(message)s',filename= path_logging + '/list_image.log');#,level=logging.info);
    for root,dirs,files in os.walk(path):
        if len(files) == 0:
            logging.fatal('No images found in "%s" directory'%(path))
            break;
            
        for file_name in files:
            if file_name.endswith(('.jpg', '.jpeg', '.JPG','.Jpg')):
                list_image.append((path + file_name));
                logging.info(file_name + ' found')
    
    logging.info('Total images found is: %s'%(len(list_image)))
    end_time = time.time();
    logging.info('Total time taken is: %s sec'%(str(end_time-start_time)))
    return list_image
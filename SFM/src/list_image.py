""" Extracting list of images from folder, get_image.py"""


def list_image(path, path_logging):

    import logging
    reload(logging)
    import time
    import os
    import sys

#    If path doesnt exist then throw error
    if os.path.exists(path):
        #             Setup logging of function
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        def setup_logger(name, log_file, level=logging.DEBUG):
            """Function setup as many loggers as you want"""

            handler = logging.FileHandler(log_file)
            handler.setFormatter(formatter)

            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(handler)

            return logger

#         Logging setup
        logger = setup_logger('get_image', os.path.join(
            path_logging, 'get_image.log'))
        logger.info('This will record message from get_image function')

#         Starting recording time
        start_time = time.time()

        list_image = []  # Getting list of images

#         Searching of image
        for root, dirs, files in os.walk(path):
            if len(files) == 0:
                logger.fatal('No images found in "%s" directory' % (path))
                break

            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.JPG', '.Jpg')):
                    list_image.append((os.path.join(path, file_name)))
                    logger.info(file_name + ' found')

        logger.info('Total images found is: %s' % (len(list_image)))
        end_time = time.time()
        logger.info('Total time taken is: %s sec' %
                    (str(round(end_time-start_time, 1))))

#         returning list of images present in the folder
        return list_image
    else:
        sys.exit("Input path provided doesn't exist")

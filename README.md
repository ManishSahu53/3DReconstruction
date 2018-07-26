# Dimension

Dimension is Indshine's Structure from Motion software. It is used to estimate
precise 3D location and orientation of images. 

## Getting Started

These instructions will get you a copy of the project up and running on your 
local machine for development and testing purposes. See deployment for notes on 
how to deploy the project on a live system.

### Prerequisites

*Dimension* depends on Python 2.7 libraries. List of libraries can be found below:
1.  [OpenCV](https://opencv.org/) Opencv is open source computer vision library.
To install opencv type ```pip install opencv-contrib-python```. This will install latest
opencv availabe. Version 3.4.1 was used to develop dimension. To download specific
version of openCV type ```pip install opencv-contrib-python==3.4.1 ```

2.  [Numpy](http://www.numpy.org/) Numpy is to store images as array of numbers.
To install numpy type ```pip install numpy```

3.  [cPickle](https://docs.python.org/2/library/pickle.html)The pickle module 
implements a fundamental, but powerful algorithm for serializing and 
de-serializing a Python object structure. This is inbuilt function in python.

4.  [Multiprocessing](https://docs.python.org/2/library/multiprocessing.html) 
This is used to do multicore processing. To install this type
```pip install multiprocessing```

5. [Argparse](https://docs.python.org/2.7/library/argparse.html)
Argparse is used to parse user input data. Install this type 
```pip install argparse```

6. [exifread](https://github.com/ianare/exif-py)
exifread is used to read exif data from images. to Install this type
```pip install ExifRead```

7. [Pyyaml](https://pyyaml.org/wiki/PyYAML)
Pyyaml is a data serialization format designed to easy readability and interaction with python.

8. [utm](https://github.com/Turbo87/utm)
Utm is used to convert coordinate system from geographic to projected and vice versa.

### Installing

To install Dimension on your PC, simply clone this repository 
``` 
git clone https://gitlab.com/manish.indshine/Dimension.git
git checkout develop
```
and install dependencies mentoned below.
To install simply type ```pip install -r requirements.txt``` or install manually as given below.
```
pip install opencv-contrib-python
pip install multiprocessing
pip install argparse
pip install exifread
pip install PyYAML
pip install utm
pip install loky
pip install six
```
Python 2.7 is used to develop so using python3 
can cause some error.

### Functions available
There are three function avaliable.

* **get_image.py**

This function is used to get list of images present in the given folder. This is 
used along with other function. To import get_image function simply type 

```import get_image```

This will load the module. It further consists list_image
function. list_image will take image directory and logging directory as inputs.
It returns list of images present inside image directory.

* **exif.py**

Exif function is used to extract exif information from the images. Two inputs are 
required - **Input directory** containing images and output directory of results.
In case no output directory is given then inside current directory output folder
will be created automatically and all the results will be save there. By default
a logging folder will be created inside current directory, this will keep the logs 
of this function.

**Results**
Result of *exif.py* will be *exif.json* file and *imagepair.json*. *Exif.json* contains information about 
position, time of capture, width, height etc of the image and *imagepair.json* contains neighbouring images name.


For example - if number of neighbouring images is 9 then first name represents master image name and rest 9 images will be 
the neighbours of the master images.

* **extract_feature.py**

Extract features function is used to extract features from images. 
There are 4 parameters available.
1. *Input directory* - Input directory will contain images
2. *Output directory* - output folder will contain extracted features
3. *Method* - Method to use for feature extraction and number of features to be extracted. It is described below
4. *Number of features* Enter number of features to be extracted. Number of features will take an integer generally around 40,000

There are 6 **Methods** available to extract features. Type 1-6 digits to use 
following methods
1. [SIFT](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
2. [SURF](https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html)
3. [ORB](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)
4. [BRISK](https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html)
5. [AKAZE](https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html)
6. [Star and BRIEF](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html)

* **Match_feature.py**

Match feature function is used to match extracted features from image pair. Image pair is obtained from *exif.py* function. There are two type of images master and pair images. Master image is the reference to which all the pair images will be matched. 

There are 7 inputs which are required to run Matching process.
1. *Input* directory containing extracted features
2. *Output* directory where all the output will be stored. (Default) is output folder in current directory
3. *Method*  that will be used to match features. Enter an integer value corresponding to options given below. Available methods are 1 -- ANN/BruteForce. ANN is used in case of SIFT and SURF features. BruteForce is used in case of Binary features. Other methods are not available yet.
4. *Parameter* is used to calculate neighbour. Available methods are 1. Euclidian Distance/Hamming Distance 2. Other
(Default) is Euclidian Distance is used for SIFT,SURF. Hamming is used for Binary decriptors. Other methods are not available yet.
5. *Features* feature method that is used for matching 1 -- SIFT 2 -- SURF 3 -- ORB 4 -- BRISK 5 -- AKAZE 6 -- STAR+ BRIEF (Default) is SIFT
6. *RATIO*, Define a ratio threshold for ratio test. (Default) is 0.8
7. *THREAD*, Number of processing threads to be used for multiprocessing. (Default) Using all the threads available.




## Running the tests

1. To run test dataset, change directory or go to **Dimension/SFM/extract_feature**. 
The test dataset contains 20 images. They are present in **Dimension/test_dataset/images** folder.

First step would be to get exif information from those images and number of neighbours are 9.
Let output folder be results directory and input be test_dataset.


```python exif.py -i ../../test_dataset/images -o output -n 9```


To get help type ```python exif.py -h``` to get information about how to run it.


This will extract exif information from images and output **exif.json** file.
This output json file contains exif information like *Latitude, Longitude, Elevation, Time of photo capture,focal length, height and width* of image in pixels.

2. Next step would be to extract features from images. In this example we will 
extract feature using SIFT algorithm. So *m* would be 1. 


``` python extract_feature.py -i ../../test_dataset/Images -o output -m 1```


To get help type ```python extract_feature.py -h``` to get information about how to run it.

There are 6 methods available so *m* can vary from 1-6.
Output folder with the name of method(here sift) will be created. **extract_feature.json** 
will be created inside logging folder to get summary of the process.


3. After extracting features next step is to match those features and then calculate fundamental matrix
In this example we will use SIFT features, method of matching is ANN, Parameter for similarity is Euclidian distance, ratio is 0.8, and number of threads on maximum available.

```python matching_feature.py -i ./output/ -o ./output/ -f 1```

To get help type ```python matching_feature.py -h````
In the output folder, matching_feature folder will be created. Data, logging, report are its subfolders containing matches, detail of individual image, and report for overall step respectively.


## To Do
1. SFM
   - Features Extraction
     * - [ ] Automatic Thresholding of different methods by assigning desired number of features
     * - [ ] Saving features with image for visualization
     * - [ ] Cluster Features extraction
     * - [ ] Setup GPU + CPU extraction
     * - [ ] Multi GPU support

   - Features Matching
     * - [ ] Find alternative to Brute Force in Binary features
     * - [ ] Matching techniques like cascade Hashing
     * - [ ] Consistency Checks for matching
     * - [ ] GPU Matching
     * - [ ] Cluster Feature Matching

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Manish Sahu** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


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

### Installing

To install Dimension on your PC, simply clone this repository and 
install dependencies mentoned above. Python 2.7 is used to develop so using python3 
can cause some error.

### Functions available
There are three function avaliable.

1. **get_image.py**

This function is used to get list of images present in the given folder. This is 
used along with other function. To import get_image function simply type 

```import get_image```

This will load the module. It further consists list_image
function. list_image will take image directory and logging directory as inputs.
It returns list of images present inside image directory.

2. **exif.py**

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

3. **extract_feature.py**

Extract features function is used to extract features from images. There are 3 
parameters available. Input directory, Output directory , 
Method to use for feature extraction. Input directory will contain images,
output folder will contain extracted features. 

There are 6 **Methods** available to extract features. Type 1-6 digits to use 
following methods
1. [SIFT](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
2. [SURF](https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html)
3. [ORB](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)
4. [BRISK](https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html)
5. [AKAZE](https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html)
6. [Star and BRIEF](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html)


## Running the tests

1. To run test dataset, go to **test_dataset**. There are 20 images available.
First step would be to get exif information from those images and number of neighbours are 9.
Let output folder be results directory and input be test_dataset.


Type ```python exif.py -i test_dataset -o output -n 9```


To get help type ```python exif.py -h``` to get information about how to run it.


This will extract exif information from images and output **exif.json** file.
This file contains exif information like *Latitude, Longitude, Elevation, Time of photo capture,
focal length, height and width* of image in pixels.

2. Next step would be to extract features from images. In this example we will 
extract feature using SIFT algorithm. So *m* would be 1. 


``` python extract_feature.py -i test_dataaset -o output -m 1```

There are 5 methods available so *m* can vary from 1-6.
Output folder with the name of method(here sift) will be created. **extract_feature.json** 
will be created inside logging folder to get summary of the process.

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


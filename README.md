# Regionprops
Regionprops is a C++ version of regionprops provided by Matlab.

## Requirements
Regionprops requires the following packeges to build:

* OpenCV (< 3.0)

## How to use
Once you compile your own project, you can use Regionprops as follows:
```c++
cv::Mat img; //input image
cv::Mat gray; //grayscale version of the input image
cv::Mat bin; //binary version of the grayscale image

//Find contours
std::vector< std::vector<cv::Point> > contours;
std::vector<cv::Vec4i> hierarchy;
cv::findContours(bin, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

//select the contours that you want, for example the contour with the biggest area
std::vector<cv::Point> contour = biggest(contours); //biggest is an invented function

RegionProps regionProps(contour, gray);
Region r = regionProps.getRegion(); //r will contain all the information about the contour
```
## Python wrapper
See src/python folder. To build the python extension cd into src/python folder, edit include folders in setup.py and type following:
```
python setup.py build_ext --inplace

```
And, run test.py in src/python

![alt text](https://github.com/aferust/regionprops/blob/master/src/python/exampleOutput.png?raw=true)
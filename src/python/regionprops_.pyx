#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
import cython

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.string cimport memcpy

cdef extern from "opencv2/core.hpp":
    cdef int CV_8UC3
    cdef int CV_8UC1
    cdef int CV_RETR_LIST
    cdef int CV_CHAIN_APPROX_NONE

cdef extern from "opencv2/core.hpp" namespace "cv":

    cdef cppclass Point:# "cv::Vec<double, 2>": 
        Point() except +
        Point(double, double) except +
        double x
        double y    
    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int, void*) except +
        void create(int, int, int)
        void* data
        int rows
        int cols
        int channels()
    cdef cppclass Point2f:
        Point2f() except +
        Point2f(double, double) except +
        double x
        double y
    cdef cppclass Size2f:
        Size2f() except +
        Size2f(double, double) except +
        double height, width
    cdef cppclass RotatedRect:
        RotatedRect () except +
        RotatedRect (const Point2f &, const Size2f &, float) except +
        RotatedRect (const Point2f &, const Point2f &, const Point2f &) except +
        float angle
        Point2f center
        Size2f size
    cdef cppclass Scalar:
        Scalar() except +
        Scalar(double) except +
        Scalar(double, double, double) except +
        Scalar(double, double, double, double) except +
        
    cdef cppclass Vec4i "cv::Vec<int, 4>": 
        Vec4i() except +
        Vec4i(double x1, double y1, double x2, double y2) except +
        Vec4i(int x1, int y1, int x2, int y2) except +
    cdef cppclass Rect "cv::Rect<int, 4>": 
        Rect() except +
        Rect(double x1, double y1, double x2, double y2) except +
        Rect(int x1, int y1, int x2, int y2) except +
        double height, width, x, y

cdef extern from "opencv2/core.hpp" namespace "cv::Moments":
    cdef cppclass Moments:
        Moments() except +
        Moments(double, double, double, double, double, double, double, double, double, double) except +
        double 	m00
        double 	m10
        double 	m01
        double 	m20
        double 	m11
        double 	m02
        double 	m30
        double 	m21
        double 	m12
        double 	m03
    
cdef extern from "opencv2/imgproc/imgproc.hpp" namespace "cv":
     void findContours(Mat mbin, vector[vector[Point]] contours, vector[Vec4i] hierarchy, int rl, int can)

cdef extern from "<valarray>" namespace "std":
    cdef cppclass valarray[T]:
        valarray()
        valarray(unsigned char)
        valarray(double)
        T& operator[](unsigned char)
        T& operator[](double)

cdef extern from "../region.h":
    cdef cppclass Region:
        Region() except +
        double Area()
        double Perimeter()
        Rect BoundingBox()
        Moments Moments()
        vector[Point] ConvexHull()
        double ConvexArea()
        RotatedRect Ellipse()
        double Orientation()
        double MinorAxis()
        double MajorAxis()
        vector[Point] Approx()
        Mat FilledImage()
        Point Centroid()
        double AspectRatio()
        double EquivalentDiameter()
        double Eccentricity()
        double FilledArea()
        Mat PixelList()
        Mat ConvexImage()
        double MaxVal()
        double MinVal()
        Point MaxLoc()
        Point MinLoc()
        Scalar MeanVal()
        vector[Point] Extrema()
        double Solidity()

cdef extern from "mwrap.h":
    Region getRInstance(const vector[Point] &_contour, const Mat &_img)
    vector[double] scalarToVector(Scalar s);

cdef np.ndarray[np.float_t, ndim=2, mode="c"] vp2np(vector[Point] p):
    cdef int i
    cdef np.ndarray[np.float_t, ndim=2, mode="c"] n2d = np.zeros((p.size(), 2), dtype=np.float)

    for i in range(p.size()):
        n2d[i,0]= p[i].x
        n2d[i,1]= p[i].y
    return n2d

cdef np.ndarray[np.float_t, ndim=2, mode="c"] mat2np(Mat mat2d):
    cdef int r, c, i, j
    r = mat2d.rows
    c = mat2d.cols
    cdef double[:] ndt =  np.empty(r*c, dtype=np.float)
    cdef int index = 0
    for i in range(r):
        for j in range(c):
            ndt[index] = <double>mat2d.data[index]
            index = index + 1
    return np.asarray(ndt).reshape((r,c))

cpdef regionprops(
                np.ndarray[np.uint8_t, ndim=2, mode="c"] mbin,
                np.ndarray[np.uint8_t, ndim=2, mode="c"] mgray
                ):
    cdef int r = mbin.shape[0]
    cdef int c = mbin.shape[1]
    cdef int i, j
    
    cdef Mat cbin
    cbin.create(r, c, CV_8UC1)
    memcpy(cbin.data, &mbin[0,0], r*c)
    
    cdef Mat cgray
    cgray.create(r, c, CV_8UC1)
    memcpy(cgray.data, &mgray[0,0], r*c)

    cdef vector[vector[Point]] contoursv;
    cdef vector[Vec4i] hierarchy;
    findContours(cbin, contoursv, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE)
    
    cdef Region _region
    cdef vector[Point] contour
    cdef vector[Point] cvhull

    regions = []
    for j in range(contoursv.size()):
        m_dict = {}

        _region = getRInstance(contoursv[j], cgray)
        
        m_dict = {
            "Area": _region.Area(),
            "Perimeter": _region.Perimeter(),
            "BoundingBox": {"x":_region.BoundingBox().x, "y":_region.BoundingBox().y, "height":_region.BoundingBox().height, "width":_region.BoundingBox().width},
            "Moments": {
                        "m00":_region.Moments().m00,
                        "m10":_region.Moments().m10,
                        "m01":_region.Moments().m01,
                        "m20":_region.Moments().m20,
                        "m11":_region.Moments().m11,
                        "m02":_region.Moments().m02,
                        "m30":_region.Moments().m30,
                        "m21":_region.Moments().m21,
                        "m12":_region.Moments().m12,
                        "m03":_region.Moments().m03
            },
            "ConvexHull": vp2np(_region.ConvexHull()),
            "ConvexArea": _region.ConvexArea(),
            "Ellipse": {"Angle":_region.Ellipse().angle,
                        "Center_x":_region.Ellipse().center.x, 
                        "Center_y":_region.Ellipse().center.y,
                        "Size_h":_region.Ellipse().size.height,
                        "Size_w":_region.Ellipse().size.width
            },
            "Orientation": _region.Orientation(),
            "MinorAxis": _region.MinorAxis(),
            "MajorAxis": _region.MajorAxis(),
            "Approx": vp2np(_region.Approx()),
            "FilledImage": mat2np(_region.FilledImage()),
            "Centroid": {"x":_region.Centroid().x, "y":_region.Centroid().y },
            "AspectRatio": _region.AspectRatio(),
            "EquivalentDiameter": _region.EquivalentDiameter(),
            "Eccentricity": _region.Eccentricity(),
            "FilledArea":  _region.FilledArea(),
            "PixelList": mat2np(_region.PixelList()),
            "ConvexImage": mat2np(_region.ConvexImage()),
            "MaxVal": _region.MaxVal(),
            "MinVal": _region.MinVal(),
            "MaxLoc": {"x": _region.MaxLoc().x, "y": _region.MaxLoc().y },
            "MinLoc": {"x": _region.MinLoc().x, "y": _region.MinLoc().y },
            "MeanVal": scalarToVector(_region.MeanVal()), # not sure if it is the best implementation
            "Extrema": vp2np(_region.Extrema()),
            "Solidity": _region.Solidity()
        }

        regions.append(m_dict)
    
    return regions


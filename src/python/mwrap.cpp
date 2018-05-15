#include <opencv2/core.hpp>
#include "../regionprops.h"
#include "../region.h"
#include <string>
#include <vector>

Region getRInstance(
                    const std::vector<cv::Point> &_contour,
                    const cv::Mat &_img){
    RegionProps regionProps(_contour, _img);
    Region r = regionProps.getRegion();
    return r;
}

std::vector<double> scalarToVector(cv::Scalar s){
    std::vector<double> v;
    for(int i = 0; i < 4; i++){
        v.push_back(s[i]);
    }
    return v;
}
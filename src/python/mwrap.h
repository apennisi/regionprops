#include <opencv2/core.hpp>
#include "../region.h"
#include <vector>

Region getRInstance(const std::vector<cv::Point> &_contour, const cv::Mat &_img);

std::vector<double> scalarToVector(cv::Scalar s);


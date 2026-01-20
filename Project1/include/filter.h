#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>

int greyscale(const cv::Mat &src, cv::Mat &dst);
int sepia(const cv::Mat &src, cv::Mat &dst);

#endif
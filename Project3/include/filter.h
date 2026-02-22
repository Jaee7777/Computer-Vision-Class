/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 3.

  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Claude AI was used for code review and debugging.

*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>

int greyscale(const cv::Mat &src, cv::Mat &dst);
int blur5x5_grey(const cv::Mat &src, cv::Mat &dst);
int saturation(const cv::Mat &src, cv::Mat &dst);
int threshold(const cv::Mat &src, cv::Mat &dst, int thresh);
int kmeansThreshold(const cv::Mat &src, int k = 2, int maxIter = 100);
int erode(const cv::Mat &src, cv::Mat &dst, int radius = 1);
int dilate(const cv::Mat &src, cv::Mat &dst, int radius = 1);
int opening(const cv::Mat &src, cv::Mat &dst, int radius = 1);
int closing(const cv::Mat &src, cv::Mat &dst, int radius = 1);

#endif
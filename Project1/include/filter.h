/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 1.

  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Claude AI was used for code review and debugging.

*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>

int greyscale(const cv::Mat &src, cv::Mat &dst);
int sepia(const cv::Mat &src, cv::Mat &dst);
int blur5x5_1(const cv::Mat &src, cv::Mat &dst);
int blur5x5_2(const cv::Mat &src, cv::Mat &dst);
int sobelX3x3(const cv::Mat &src, cv::Mat &dst );
int sobelY3x3(const cv::Mat &src, cv::Mat &dst );

#endif
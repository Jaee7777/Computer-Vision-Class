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

#ifndef UTIL_H
#define UTIL_H

#include <opencv2/core.hpp>

int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug );
void prepEmbeddingImage( cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta, float minE1, float maxE1, float minE2, float maxE2, int debug );

#endif
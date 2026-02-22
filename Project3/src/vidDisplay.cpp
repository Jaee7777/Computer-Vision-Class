/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 3.

  Skeleton code provided on the assignment page (Project 1) was used as a template.

  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Claude AI was used for code review and debugging.

*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "../include/filter.h"
#include "../include/faceDetect.h"
#include "../include/DA2Network.hpp"

int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        // initialization
        cv::Mat frame, sat, blurred, thresh, binary_erode, binary_dilate, binary_open, binary_close;
        int threshVal = 80;
        int frameCount = 0;

        cv::Mat labels, stats, centroids;

        // Initialize DA2Network for depth estimation
        // DA2Network da_net("model/model_fp16.onnx");

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if (frame.empty()) {
                  printf("frame is empty\n");
                  break;
                }   

                saturation(frame, sat);
                blur5x5_grey(sat, blurred);
                threshVal = kmeansThreshold(blurred, 2); // Using k=2
                threshold(blurred, thresh, threshVal);
                erode(thresh, binary_erode, 1); // Using radius=1 for 3x3 kernels.
                dilate(thresh, binary_dilate, 1);
                opening(thresh, binary_open, 1);
                closing(thresh, binary_close, 1);

                int numComponents = cv::connectedComponentsWithStats(thresh, labels, stats, centroids, 8);

                for (int i = 1; i < numComponents; ++i) {
                    if (stats.at<int>(i, cv::CC_STAT_AREA) < 500) continue;

                    int x = stats.at<int>(i, cv::CC_STAT_LEFT);
                    int y = stats.at<int>(i, cv::CC_STAT_TOP);
                    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
                    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                    double cx = centroids.at<double>(i, 0);
                    double cy = centroids.at<double>(i, 1);

                    cv::rectangle(frame, cv::Point(x, y), cv::Point(x+w, y+h),
                                cv::Scalar(0, 255, 0), 2);          // green box for region
                    cv::circle(frame, cv::Point((int)cx, (int)cy), 5,
                            cv::Scalar(0, 0, 255), -1);            // red dot for centroid
                    cv::putText(frame, std::to_string(i),             // component ID label
                                cv::Point(x, y - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 255, 0), 1);
                }

                cv::imshow("Original", frame);
                cv::imshow("Saturation", sat);
                cv::imshow("Thresholded", thresh);
                // cv::imshow("Eroded", binary_erode);
                // cv::imshow("Dilated", binary_dilate);
                // cv::imshow("Opening", binary_open);
                cv::imshow("Closing", binary_close);
                cv::imshow("Detections", frame);
                
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);

                if( key == 'q') {
                    break;
                }
                else if (key == 's') {
                    std::string filename = "frame_thresh_" + std::to_string(frameCount) + ".jpg";
                    cv::imwrite(filename, thresh);
                    frameCount++;
                }
        }

        delete capdev;
        return(0);
}

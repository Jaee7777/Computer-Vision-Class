/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 1.

  Skeleton code provided on the assignment page was used as a template.

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

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        int frameCount = 0;

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }                
                cv::imshow("Video", frame);

                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                if( key == 'q') {
                    break;
                }
                else if (key == 's') {
                    std::string filename = "frame_" + std::to_string(frameCount) + ".jpg";
                    cv::imwrite(filename, frame);
                    frameCount++;
                }
                else if (key == 'g') {
                    cv::Mat greyFrame;
                    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                    cv::imshow("Greyscale Image", greyFrame);
                }
                else if (key == 'h') {
                    cv::Mat greyFrame_custom;
                    greyscale(frame, greyFrame_custom);
                    cv::imshow("Greyscale Image - Average", greyFrame_custom);
                }
                else if (key == 'j') {
                    cv::Mat sepiaFrame;
                    sepia(frame, sepiaFrame);
                    cv::imshow("Sepia Image", sepiaFrame);
                }
                else if (key == 'b') {
                    cv::Mat blurFrame;
                    blur5x5_2(frame, blurFrame);
                    cv::imshow("Blurred Image", blurFrame);
                }
                else if (key == 'x') {
                    cv::Mat sobelXFrame, displayX;
                    sobelX3x3(frame, sobelXFrame);
                    cv::convertScaleAbs(sobelXFrame, displayX); // Convert to 8-bit for display.
                    cv::imshow("Sobel X Image", displayX);
                }
                else if (key == 'y') {
                    cv::Mat sobelYFrame, displayY;
                    sobelY3x3(frame, sobelYFrame);
                    cv::convertScaleAbs(sobelYFrame, displayY); // Convert to 8-bit for display.
                    cv::imshow("Sobel Y Image", displayY);
                }
        }

        delete capdev;
        return(0);
}

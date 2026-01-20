/*
Jaee Oh

CS5330 Computer Vision Project 1: Task 2

Using skeleton code from the assignment description.
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
        }

        delete capdev;
        return(0);
}

/*
Jaee Oh

CS5330 Computer Vision Project 1: Task 1

Referenced from OpenCV Tutorials (https://docs.opencv.org/4.5.1/db/deb/tutorial_display_image.html)
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "../include/filter.h"

using namespace cv;

int main() {
    Mat image = imread("texture_draw.png", IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Could not read the image!" << std::endl;
        return 1;
    }

    imshow("Display window", image);

    while (true) {
        int k = waitKey(0);

        if (k == 's') {
            imwrite("cat_copy.png", image);
        }
        else if (k == 'q') {
            break;
        }
        else if (k == 'm') {
            cv::Mat sobelXFrame, sobelYFrame, magnitudeFrame;
            sobelX3x3(image, sobelXFrame);
            sobelY3x3(image, sobelYFrame);
            magnitude(sobelXFrame, sobelYFrame, magnitudeFrame);
            cv::imshow("Sobel Magnitude Image", magnitudeFrame);
        }
    }

    return 0;
}
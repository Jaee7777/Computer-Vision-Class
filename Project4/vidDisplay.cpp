/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 4.

  Skeleton code provided on the assignment page (Project 1) was used as a template.

  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Claude AI was used for code review and debugging.

*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    // Open video camera
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    cv::Size refS(
        (int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    std::cout << "Expected size: "
              << refS.width << " "
              << refS.height << std::endl;

    cv::Mat frame;
    std::vector<cv::Point2f> corner_set;

    // Find checkboard of 10x7 corners
    cv::Size pattern_size(9, 6);

    bool last_found = false;

    while (true)
    {
        capdev >> frame;
        if (frame.empty()) {
            std::cout << "Frame is empty\n";
            break;
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Clear corners from previous frame
        corner_set.clear();

        bool found = cv::findChessboardCorners(
            gray,
            pattern_size,
            corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
        );

        // Display when checkboard detected or lost
        if (found != last_found) {
            std::cout << (found ? "Checkerboard detected" : "Checkerboard lost") << std::endl;
            last_found = found;
        }

        if (found) {
            cv::cornerSubPix(
                gray,
                corner_set,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(
                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                    30,
                    0.1
                )
            );

            // Draw corners on frame, not on grayscale
            cv::drawChessboardCorners(
                frame,
                pattern_size,
                corner_set,
                found
            );

            if (!corner_set.empty()) {
                std::cout << "Corners: " << corner_set.size()
                          << "  First: ("
                          << corner_set[0].x << ", "
                          << corner_set[0].y << ")"
                          << std::endl;
            }
        }

        cv::imshow("Video", frame);

        int key = cv::waitKey(10);
        if (key == 'q')
            break;
    }

    return 0;
}
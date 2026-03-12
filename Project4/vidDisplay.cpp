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

    std::vector<std::vector<cv::Point2f>> corner_list; // 2D corners
    std::vector<std::vector<cv::Point3f>> point_list; // 3D points
    int calib_count = 0; 

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
        if (key == 'q') { // Quit
            break;
        } else if (key == 's') { // Save calibration frame
            if (found && !corner_set.empty()) { // Save when checkboard corners found
                corner_list.push_back(corner_set); // Save 2D coordinates of corners

                std::vector<cv::Point3f> point_set;
                for (int r = 0; r < pattern_size.height; r++) {
                    for (int c = 0; c < pattern_size.width; c++) {
                        point_set.push_back(cv::Point3f((float)c, (float)-r, 0.0f));
                    }
                }
                point_list.push_back(point_set); // Save 3D coordinates of corners (Z=0 is checkerboard plane)

                // Save calibration image
                std::string fname = "calib_" + std::to_string(calib_count++) + ".jpg";
                cv::imwrite(fname, frame);

                std::cout << "Saved calibration frame: " << fname
                          << "  (total: " << corner_list.size() << ")" << std::endl;
            } else {
                std::cout << "Not saved, because checkerboard not detected." << std::endl;
            }
        } else if (key == 'c') { // Calibrate camera
            if (corner_list.size() < 5) {
                std::cout << "Need at least 5 calibration frames, but given: "
                        << corner_list.size() << std::endl;
            } else {
                // Camera matrix initialization
                cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                camera_matrix.at<double>(0, 0) = 1;  // fx guess
                camera_matrix.at<double>(1, 1) = 1;  // fy guess
                camera_matrix.at<double>(0, 2) = frame.cols / 2.0;  // cx = image center
                camera_matrix.at<double>(1, 2) = frame.rows / 2.0;  // cy = image center

                cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

                std::vector<cv::Mat> rvecs, tvecs;

                double rms = cv::calibrateCamera(
                    point_list,
                    corner_list,
                    frame.size(),
                    camera_matrix,
                    dist_coeffs,
                    rvecs,
                    tvecs,
                    cv::CALIB_FIX_ASPECT_RATIO
                );

                std::cout << "Calibration RMS error: " << rms << std::endl;
                std::cout << "Camera matrix:\n" << camera_matrix << std::endl;
                std::cout << "Distortion coefficients:\n" << dist_coeffs << std::endl;

                // Save calibration results to file
                cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
                fs << "camera_matrix" << camera_matrix;
                fs << "dist_coeffs" << dist_coeffs;
                fs.release();
                std::cout << "Saved to calibration.yml" << std::endl;
            }
        }
    }

    return 0;
}
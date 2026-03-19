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

// ============================================
// Draw a 3D cube on top of the checkerboard
// ============================================
void drawCube(
    cv::Mat &frame,
    const std::vector<cv::Point2f> &corner_set,
    const cv::Size &pattern_size,
    const float square_size_mm,
    const cv::Mat &camera_matrix,
    const cv::Mat &dist_coeffs)
{
    // Build 3D world grid
    std::vector<cv::Point3f> point_set;
    for (int r = 0; r < pattern_size.height; r++)
        for (int c = 0; c < pattern_size.width; c++)
            point_set.push_back(cv::Point3f(
                 c * square_size_mm,
                -r * square_size_mm,
                 0.0f
            ));

    cv::Mat rvec, tvec;
    cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

    // Cube side length = 3 squares, offset to center of board
    float s  = 3 * square_size_mm;
    float ox = 3 * square_size_mm;  // X offset: 3 squares right
    float oy = -3 * square_size_mm; // Y offset: 3 squares down

    std::vector<cv::Point3f> cube_3d = {
        {ox,     oy,     0 },  // 0 bottom-front-left
        {ox + s, oy,     0 },  // 1 bottom-front-right
        {ox + s, oy - s, 0 },  // 2 bottom-back-right
        {ox,     oy - s, 0 },  // 3 bottom-back-left
        {ox,     oy,    -s },  // 4 top-front-left
        {ox + s, oy,    -s },  // 5 top-front-right
        {ox + s, oy - s,-s },  // 6 top-back-right
        {ox,     oy - s,-s },  // 7 top-back-left
    };

    // Project all 8 corners to 2D pixel positions
    std::vector<cv::Point2f> cube_2d;
    cv::projectPoints(cube_3d, rvec, tvec, camera_matrix, dist_coeffs, cube_2d);

    // Convert Point2f to Point for fillConvexPoly
    auto p = [&](int i) { return cv::Point(cube_2d[i]); };

    // Top face — blue
    std::vector<cv::Point> top_face    = { p(4), p(5), p(6), p(7) };
    cv::fillConvexPoly(frame, top_face, {255, 100, 0});

    // Front face — green
    std::vector<cv::Point> front_face  = { p(0), p(1), p(5), p(4) };
    cv::fillConvexPoly(frame, front_face, {0, 200, 80});

    // Side face — red
    std::vector<cv::Point> side_face   = { p(1), p(2), p(6), p(5) };
    cv::fillConvexPoly(frame, side_face, {0, 60, 220});

    // Bottom face edges
    cv::line(frame, cube_2d[0], cube_2d[1], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[1], cube_2d[2], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[2], cube_2d[3], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[3], cube_2d[0], {50, 50, 50}, 2);

    // Top face edges
    cv::line(frame, cube_2d[4], cube_2d[5], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[5], cube_2d[6], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[6], cube_2d[7], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[7], cube_2d[4], {50, 50, 50}, 2);

    // Vertical edges connecting bottom to top
    cv::line(frame, cube_2d[0], cube_2d[4], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[1], cube_2d[5], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[2], cube_2d[6], {50, 50, 50}, 2);
    cv::line(frame, cube_2d[3], cube_2d[7], {50, 50, 50}, 2);
}

// ============================================
// Draw Harris corners on the frame
// ============================================
void drawHarris(cv::Mat &frame, const cv::Mat &gray)
{
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);

    cv::Mat harris_response;
    cv::cornerHarris(
        gray_f,
        harris_response,
        2,     // block size: neighborhood around each pixel
        3,     // Sobel kernel size: used to compute gradients
        0.04   // Harris free parameter k
    );

    // Normalize response to 0-255 for easy thresholding
    cv::Mat harris_norm;
    cv::normalize(harris_response, harris_norm, 0, 255, cv::NORM_MINMAX, CV_32F);

    // Mark pixels above threshold as corners
    for (int r = 0; r < harris_norm.rows; r++)
        for (int c = 0; c < harris_norm.cols; c++)
            if (harris_norm.at<float>(r, c) > 150.f)
                cv::circle(frame, {c, r}, 4, {0, 0, 255}, -1); // red dot

    cv::putText(frame, "Harris ON", {10, 30},
        cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
}

// ============================================
// Draw 3D axes on the checkerboard
// ============================================
void drawAxes(
    cv::Mat &frame,
    const std::vector<cv::Point2f> &corner_set,
    const cv::Size &pattern_size,
    const float square_size_mm,
    const cv::Mat &camera_matrix,
    const cv::Mat &dist_coeffs)
{
    // Build 3D world grid matching the detected corners
    std::vector<cv::Point3f> point_set;
    for (int r = 0; r < pattern_size.height; r++)
        for (int c = 0; c < pattern_size.width; c++)
            point_set.push_back(cv::Point3f(
                 c * square_size_mm,
                -r * square_size_mm,
                 0.0f
            ));

    // Solve for board pose: rvec = rotation, tvec = translation
    cv::Mat rvec, tvec;
    cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

    // Print pose to console
    std::cout << "--- Board Pose ---" << std::endl;
    std::cout << "Translation (mm) x: " << tvec.at<double>(0)
              << "  y: "                << tvec.at<double>(1)
              << "  z: "                << tvec.at<double>(2) << std::endl;
    cv::Mat rot_matrix;
    cv::Rodrigues(rvec, rot_matrix);
    std::cout << "Rotation matrix:\n" << rot_matrix << std::endl;

    // Define axis endpoints 3 squares long in world space
    float len = 3 * square_size_mm;
    std::vector<cv::Point3f> axis_3d = {
        {0,    0,    0   },  // origin
        {len,  0,    0   },  // X tip
        {0,   -len,  0   },  // Y tip
        {0,    0,   -len }   // Z tip (toward camera)
    };

    // Project 3D axis points to 2D pixel positions
    std::vector<cv::Point2f> axis_2d;
    cv::projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs, axis_2d);

    // Draw colored axes from origin to each tip
    cv::line(frame, axis_2d[0], axis_2d[1], {0,   0,   255}, 3); // X = red
    cv::line(frame, axis_2d[0], axis_2d[2], {0,   255, 0  }, 3); // Y = green
    cv::line(frame, axis_2d[0], axis_2d[3], {255, 0,   0  }, 3); // Z = blue

    // Label each axis tip
    cv::putText(frame, "X", axis_2d[1], cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,   0,   255}, 2);
    cv::putText(frame, "Y", axis_2d[2], cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,   255, 0  }, 2);
    cv::putText(frame, "Z", axis_2d[3], cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 0,   0  }, 2);
}

int main(int argc, char *argv[])
{
    // ============================================
    // Open video camera
    // ============================================
    cv::VideoCapture capdev(0);
    // cv::VideoCapture capdev(0, cv::CAP_V4L2);
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

    // ============================================
    // Data structure definitions
    // ============================================
    cv::Mat frame;
    std::vector<cv::Point2f> corner_set;
    bool last_found = false;

    // Find checkerboard of 10x7 corners
    cv::Size pattern_size(9, 6);
    const float square_size_mm = 25.0f;  // Physical size of one square in mm

    std::vector<std::vector<cv::Point2f>> corner_list;  // 2D corners
    std::vector<std::vector<cv::Point3f>> point_list;   // 3D points
    int calib_count = 0;

    cv::Mat camera_matrix, dist_coeffs;
    bool calibrated = false;

    // Check calibration file
    cv::FileStorage fs_in("calibration.yml", cv::FileStorage::READ);
    if (fs_in.isOpened()) {
        fs_in["camera_matrix"] >> camera_matrix;
        fs_in["dist_coeffs"]   >> dist_coeffs;
        fs_in.release();
        calibrated = true;
        std::cout << "Loaded calibration from calibration.yml" << std::endl;
    } else {
        std::cout << "No calibration file found, starting with uncalibrated camera." << std::endl;
    }

    // Harris  flags
    bool show_harris = false;

    // ============================================
    // Enter Loop for Video Display and Processing
    // ============================================
    while (true)
    {
        capdev >> frame;
        if (frame.empty()) {
            std::cout << "Frame is empty\n";
            break;
        }

        // Convert to grayscale once
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ============================================
        // Harris corner overlay (toggle with 'h')
        // ============================================
        if (show_harris)
            drawHarris(frame, gray);


        // Clear corners from previous frame
        corner_set.clear();

        // ============================================
        // Checkerboard detection
        // ============================================
        bool found = cv::findChessboardCorners(
            gray,
            pattern_size,
            corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
        );

        // Print when checkerboard detected or lost
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

            // Draw corners on frame
            cv::drawChessboardCorners(frame, pattern_size, corner_set, found);

            if (!corner_set.empty()) {
                std::cout << "Corners: " << corner_set.size()
                          << "  First: ("
                          << corner_set[0].x << ", "
                          << corner_set[0].y << ")"
                          << std::endl;
            }

            // ============================================
            // AR Axes overlay (requires calibration)
            // ============================================
            if (calibrated) {
                drawAxes(frame, corner_set, pattern_size, square_size_mm,
                         camera_matrix, dist_coeffs);

                // ============================================
                // Virtual cube on the board
                // ============================================
                drawCube(frame, corner_set, pattern_size, square_size_mm,
                         camera_matrix, dist_coeffs);
            }
        }

        cv::imshow("Video", frame);

        // ============================================
        // Key handling
        // ============================================
        int key = cv::waitKey(10);

        if (key == 'q') {
            // Quit
            break;

        } else if (key == 'h') {
            // Toggle Harris corner overlay
            show_harris = !show_harris;
            std::cout << "Harris: " << (show_harris ? "ON" : "OFF") << std::endl;

        } else if (key == 's') {
            // Save calibration frame
            if (found && !corner_set.empty()) {
                corner_list.push_back(corner_set);

                std::vector<cv::Point3f> point_set;
                for (int r = 0; r < pattern_size.height; r++) {
                    for (int c = 0; c < pattern_size.width; c++) {
                        point_set.push_back(cv::Point3f(
                            c * square_size_mm,
                           -r * square_size_mm,
                            0.0f
                        ));
                    }
                }
                point_list.push_back(point_set);

                std::string fname = "calib_" + std::to_string(calib_count++) + ".jpg";
                cv::imwrite(fname, frame);
                std::cout << "Saved calibration frame: " << fname
                          << "  (total: " << corner_list.size() << ")" << std::endl;
            } else {
                std::cout << "Not saved, because checkerboard not detected." << std::endl;
            }

        } else if (key == 'c') {
            // Calibrate camera
            if (corner_list.size() < 5) {
                std::cout << "Need at least 5 calibration frames, but given: "
                          << corner_list.size() << std::endl;
            } else {
                camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                camera_matrix.at<double>(0, 2) = frame.cols / 2.0;
                camera_matrix.at<double>(1, 2) = frame.rows / 2.0;
                dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

                std::cout << "Before Calibration:" << std::endl;
                std::cout << "Camera matrix:\n"           << camera_matrix << std::endl;
                std::cout << "Distortion coefficients:\n" << dist_coeffs   << std::endl;

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

                std::cout << "After Calibration:" << std::endl;
                std::cout << "Calibration RMS error: "    << rms           << std::endl;
                std::cout << "Camera matrix:\n"           << camera_matrix << std::endl;
                std::cout << "Distortion coefficients:\n" << dist_coeffs   << std::endl;

                cv::FileStorage fs_out("calibration.yml", cv::FileStorage::WRITE);
                fs_out << "camera_matrix" << camera_matrix;
                fs_out << "dist_coeffs"   << dist_coeffs;
                fs_out.release();

                calibrated = true;
                std::cout << "Saved to calibration.yml" << std::endl;
            }
        }
    }

    return 0;
}
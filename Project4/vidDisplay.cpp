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
// Draw a simple RGB cube on the checkerboard
//   - Front face : Red
//   - Right face : Green
//   - Top face   : Blue
// ============================================
void drawCube(
    cv::Mat &frame,
    const std::vector<cv::Point2f> &corner_set,
    const cv::Size &pattern_size,
    const float square_size_mm,
    const cv::Mat &camera_matrix,
    const cv::Mat &dist_coeffs)
{
    // Build 3D world grid matching detected corners
    std::vector<cv::Point3f> point_set;
    for (int r = 0; r < pattern_size.height; r++)
        for (int c = 0; c < pattern_size.width; c++)
            point_set.push_back(cv::Point3f(
                 c * square_size_mm,
                -r * square_size_mm,
                 0.0f));

    cv::Mat rvec, tvec;
    cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

    // Cube: 3 squares per side, anchored at board origin (top-left corner)
    float s  = 3.0f * square_size_mm;
    float ox = 3.0f * square_size_mm;   // shift right
    float oy = -3.0f * square_size_mm;  // shift down (negative Y = board downward)

    std::vector<cv::Point3f> cube_3d = {
        {ox,     oy,      0},  // 0 bottom-front-left
        {ox + s, oy,      0},  // 1 bottom-front-right
        {ox + s, oy - s,  0},  // 2 bottom-back-right
        {ox,     oy - s,  0},  // 3 bottom-back-left
        {ox,     oy,     -s},  // 4 top-front-left
        {ox + s, oy,     -s},  // 5 top-front-right
        {ox + s, oy - s, -s},  // 6 top-back-right
        {ox,     oy - s, -s},  // 7 top-back-left
    };

    std::vector<cv::Point2f> cube_2d;
    cv::projectPoints(cube_3d, rvec, tvec, camera_matrix, dist_coeffs, cube_2d);

    auto p = [&](int i) { return cv::Point(cube_2d[i]); };

    // Front face — Red
    std::vector<cv::Point> front_face = {p(0), p(1), p(5), p(4)};
    cv::fillConvexPoly(frame, front_face, {0, 0, 220});

    // Right face — Green
    std::vector<cv::Point> right_face = {p(1), p(2), p(6), p(5)};
    cv::fillConvexPoly(frame, right_face, {0, 200, 0});

    // Top face — Blue
    std::vector<cv::Point> top_face = {p(4), p(5), p(6), p(7)};
    cv::fillConvexPoly(frame, top_face, {220, 0, 0});

    // --- Edges over all faces for clean outline ---
    const cv::Scalar edge_color = {30, 30, 30};
    const int edge_w = 2;

    // Bottom ring
    cv::line(frame, p(0), p(1), edge_color, edge_w);
    cv::line(frame, p(1), p(2), edge_color, edge_w);
    cv::line(frame, p(2), p(3), edge_color, edge_w);
    cv::line(frame, p(3), p(0), edge_color, edge_w);
    // Top ring
    cv::line(frame, p(4), p(5), edge_color, edge_w);
    cv::line(frame, p(5), p(6), edge_color, edge_w);
    cv::line(frame, p(6), p(7), edge_color, edge_w);
    cv::line(frame, p(7), p(4), edge_color, edge_w);
    // Vertical edges
    cv::line(frame, p(0), p(4), edge_color, edge_w);
    cv::line(frame, p(1), p(5), edge_color, edge_w);
    cv::line(frame, p(2), p(6), edge_color, edge_w);
    cv::line(frame, p(3), p(7), edge_color, edge_w);
}

// ============================================
// Draw Harris corners on the frame
// ============================================
void drawHarris(cv::Mat &frame, const cv::Mat &gray)
{
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);

    cv::Mat harris_response;
    cv::cornerHarris(gray_f, harris_response, 2, 3, 0.04);

    cv::Mat harris_norm;
    cv::normalize(harris_response, harris_norm, 0, 255, cv::NORM_MINMAX, CV_32F);

    for (int r = 0; r < harris_norm.rows; r++)
        for (int c = 0; c < harris_norm.cols; c++)
            if (harris_norm.at<float>(r, c) > 150.f)
                cv::circle(frame, {c, r}, 4, {0, 0, 255}, -1);

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
    std::vector<cv::Point3f> point_set;
    for (int r = 0; r < pattern_size.height; r++)
        for (int c = 0; c < pattern_size.width; c++)
            point_set.push_back(cv::Point3f(
                 c * square_size_mm,
                -r * square_size_mm,
                 0.0f));

    cv::Mat rvec, tvec;
    cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

    std::cout << "--- Board Pose ---" << std::endl;
    std::cout << "Translation (mm) x: " << tvec.at<double>(0)
              << "  y: "                 << tvec.at<double>(1)
              << "  z: "                 << tvec.at<double>(2) << std::endl;
    cv::Mat rot_matrix;
    cv::Rodrigues(rvec, rot_matrix);
    std::cout << "Rotation matrix:\n" << rot_matrix << std::endl;

    float len = 3.0f * square_size_mm;
    std::vector<cv::Point3f> axis_3d = {
        {0,    0,    0   },
        {len,  0,    0   },
        {0,   -len,  0   },
        {0,    0,   -len }
    };

    std::vector<cv::Point2f> axis_2d;
    cv::projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs, axis_2d);

    cv::line(frame, axis_2d[0], axis_2d[1], {0,   0,   255}, 3); // X = red
    cv::line(frame, axis_2d[0], axis_2d[2], {0,   255, 0  }, 3); // Y = green
    cv::line(frame, axis_2d[0], axis_2d[3], {255, 0,   0  }, 3); // Z = blue

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
    if (!capdev.isOpened()) {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    cv::Size refS(
        (int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT)
    );
    std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

    // ============================================
    // State
    // ============================================
    cv::Mat frame;
    std::vector<cv::Point2f> corner_set;
    bool last_found = false;

    cv::Size pattern_size(9, 6);
    const float square_size_mm = 25.0f;

    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Point3f>> point_list;
    int calib_count = 0;

    cv::Mat camera_matrix, dist_coeffs;
    bool calibrated = false;

    // Load calibration if available
    cv::FileStorage fs_in("calibration.yml", cv::FileStorage::READ);
    if (fs_in.isOpened()) {
        fs_in["camera_matrix"] >> camera_matrix;
        fs_in["dist_coeffs"]   >> dist_coeffs;
        fs_in.release();
        calibrated = true;
        std::cout << "Loaded calibration from calibration.yml" << std::endl;
    } else {
        std::cout << "No calibration file found, starting uncalibrated." << std::endl;
    }

    bool show_harris = false;
    bool show_cube   = true;   // NEW: toggle virtual cube with 'v'

    // ============================================
    // Main loop
    // ============================================
    while (true)
    {
        capdev >> frame;
        if (frame.empty()) {
            std::cout << "Frame is empty\n";
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (show_harris)
            drawHarris(frame, gray);

        corner_set.clear();

        bool found = cv::findChessboardCorners(
            gray, pattern_size, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found != last_found) {
            std::cout << (found ? "Checkerboard detected" : "Checkerboard lost") << std::endl;
            last_found = found;
        }

        if (found) {
            cv::cornerSubPix(
                gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            cv::drawChessboardCorners(frame, pattern_size, corner_set, found);

            if (!corner_set.empty())
                std::cout << "Corners: " << corner_set.size()
                          << "  First: (" << corner_set[0].x << ", " << corner_set[0].y << ")"
                          << std::endl;

            if (calibrated) {
                drawAxes(frame, corner_set, pattern_size, square_size_mm,
                         camera_matrix, dist_coeffs);

                if (show_cube)
                    drawCube(frame, corner_set, pattern_size, square_size_mm,
                             camera_matrix, dist_coeffs);
            }
        }

        cv::imshow("Video", frame);

        int key = cv::waitKey(10);

        if (key == 'q') {
            break;

        } else if (key == 'h') {
            show_harris = !show_harris;
            std::cout << "Harris: " << (show_harris ? "ON" : "OFF") << std::endl;

        } else if (key == 'v') {
            // NEW: toggle virtual cube
            show_cube = !show_cube;
            std::cout << "Virtual cube: " << (show_cube ? "ON" : "OFF") << std::endl;

        } else if (key == 's') {
            // Save a calibration frame: corners drawn, no axes/cube
            if (found && !corner_set.empty()) {
                // Make a clean copy with only corners highlighted
                cv::Mat corner_frame = frame.clone();

                cv::Mat corners_only;
                cv::cvtColor(gray, corners_only, cv::COLOR_GRAY2BGR);
                cv::drawChessboardCorners(corners_only, pattern_size, corner_set, found);

                std::string fname = "calib_" + std::to_string(calib_count) + ".jpg";
                cv::imwrite(fname, corners_only);
                std::cout << "Saved corners-only calibration image: " << fname << std::endl;

                // Store data for calibration
                corner_list.push_back(corner_set);
                std::vector<cv::Point3f> point_set;
                for (int r = 0; r < pattern_size.height; r++)
                    for (int c = 0; c < pattern_size.width; c++)
                        point_set.push_back(cv::Point3f(
                            c * square_size_mm, -r * square_size_mm, 0.0f));
                point_list.push_back(point_set);

                calib_count++;
                std::cout << "Total calibration frames: " << corner_list.size() << std::endl;
            } else {
                std::cout << "Not saved — checkerboard not detected." << std::endl;
            }

        } else if (key == 'a') {
            // NEW: save current frame with axes (and cube if enabled)
            if (found && calibrated) {
                std::string fname = "axes_" + std::to_string(calib_count) + ".jpg";
                cv::imwrite(fname, frame);
                std::cout << "Saved axes/AR frame: " << fname << std::endl;
            } else {
                std::cout << "Not saved — need checkerboard detected and calibration loaded." << std::endl;
            }

        } else if (key == 'c') {
            if (corner_list.size() < 5) {
                std::cout << "Need at least 5 calibration frames, have: "
                          << corner_list.size() << std::endl;
            } else {
                camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                camera_matrix.at<double>(0, 2) = frame.cols / 2.0;
                camera_matrix.at<double>(1, 2) = frame.rows / 2.0;
                dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

                std::cout << "Before Calibration:\n" << camera_matrix << std::endl;

                std::vector<cv::Mat> rvecs, tvecs;
                double rms = cv::calibrateCamera(
                    point_list, corner_list, frame.size(),
                    camera_matrix, dist_coeffs, rvecs, tvecs,
                    cv::CALIB_FIX_ASPECT_RATIO);

                std::cout << "RMS error: "           << rms           << std::endl;
                std::cout << "Camera matrix:\n"      << camera_matrix << std::endl;
                std::cout << "Dist coefficients:\n"  << dist_coeffs   << std::endl;

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
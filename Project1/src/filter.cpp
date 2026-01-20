#include "../include/filter.h"

int greyscale(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_8UC1); // Create cv::Mat for greyscale image. It uses 1 channel 0-255.

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j); // Access RGB values of pixel[i, j].
            dst.at<uchar>(i, j) = static_cast<uchar>(pixel[2] + pixel[1] + pixel[0])/3; // Resultant greyscale value on pixel[i, j].
        }
    }
    return 0;
}

int sepia(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), src.type()); // Create cv::Mat for sepia image. It uses 3 channels RGB.

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j); // Access RGB values of pixel[i, j].

            int tb = static_cast<int>(0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0]); // Channel order is in BGR.
            int tg = static_cast<int>(0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0]);
            int tr = static_cast<int>(0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0]);

            dst.at<cv::Vec3b>(i, j)[2] = (tr > 255) ? 255 : tr; // Make sure values are 0-255
            dst.at<cv::Vec3b>(i, j)[1] = (tg > 255) ? 255 : tg;
            dst.at<cv::Vec3b>(i, j)[0] = (tb > 255) ? 255 : tb;
        }
    }
    return 0;
}
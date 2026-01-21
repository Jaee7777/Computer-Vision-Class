/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 1.

  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Claude AI was used for code review and debugging.

*/


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

int blur5x5_1(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    src.copyTo(dst); // Initialize dst with src to keep 

    std::vector<std::vector<int>> kernel;
    kernel = { {1, 2, 4, 2, 1},
               {2, 4, 8, 4, 2},
               {4, 8, 16, 8, 4},
               {2, 4, 8, 4, 2},
               {1, 2, 4, 2, 1} };

    // Sum of kernel values
    int sum_kernel = 0;
    // Loop through each row of kernel.
    for (const auto& row : kernel) {
        // Loop through each value in the row.
        for (const auto& val : row) {
            sum_kernel += val;
        }
    }

    // Loop through each pixel in the image.
    for (int i = 2; i < src.rows - 2; ++i) {
        for (int j = 2; j < src.cols - 2; ++j) {
            int sum_b = 0, sum_g = 0, sum_r = 0;
            // Loop through the kernel centered at pixel [i, j] for all 3 RGB channels.
            for (int k = 0; k < 5; ++k) {
                for (int l = 0; l < 5; ++l) {
                    sum_b += kernel[k][l] * src.at<cv::Vec3b>(i + k - 2, j + l - 2)[0];
                    sum_g += kernel[k][l] * src.at<cv::Vec3b>(i + k - 2, j + l - 2)[1];
                    sum_r += kernel[k][l] * src.at<cv::Vec3b>(i + k - 2, j + l - 2)[2];
                }
            }
            // Normalize the sum of kernal applied to each channels.
            dst.at<cv::Vec3b>(i, j)[0] = sum_b / sum_kernel;
            dst.at<cv::Vec3b>(i, j)[1] = sum_g / sum_kernel;
            dst.at<cv::Vec3b>(i, j)[2] = sum_r / sum_kernel;
        }
    }

    return 0;
}

int blur5x5_2(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    src.copyTo(dst); // Initialize dst with src to keep border pixels.

    std::vector<int> kernel = {1, 2, 4, 2, 1};

    // Sum of kernel values
    int sum_kernel = 0;
    for (int val : kernel) {
        sum_kernel += val;
    }
    int divisor = sum_kernel * sum_kernel; // Normalization factor to apply at the end.

    cv::Mat temp(src.rows, src.cols, CV_32SC3); // 3 channel integer type that can go above 255 during calculation.

    // Horizontal blur first.
    // Loop through each row in the image.
    for (int i = 0; i < src.rows; ++i) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i); // Pointer to the input row.
        cv::Vec3i* tempRow = temp.ptr<cv::Vec3i>(i); // Keep int type during calculation. It can go above 255.

        // Loop through each column in the image.
        for (int j = 2; j < src.cols - 2; ++j) {
            int sum_b = 0, sum_g = 0, sum_r = 0;

            // Loop through horizontal kernel centered at pixel [i, j], so [i, j-2] to [i, j+2].
            for (int k = 0; k < 5; ++k) {
                int idx = (j + k - 2); // Define index for pointer of the selected row.
                sum_b += kernel[k] * srcRow[idx][0]; // Apply kernel to each channel.
                sum_g += kernel[k] * srcRow[idx][1];
                sum_r += kernel[k] * srcRow[idx][2];
            }
            
            tempRow[j][0] = sum_b; // Save unnormalized values to temp.
            tempRow[j][1] = sum_g;
            tempRow[j][2] = sum_r;
        }

    }

    // Vertical blur.
    // Loop through each row in the image.
    for (int i = 2; i < src.rows - 2; ++i) {
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i); // Pointer to the output row.

        // Loop through each column in the image.
        for (int j = 2; j < src.cols - 2; ++j) {
            int sum_b = 0, sum_g = 0, sum_r = 0;

            // Loop through vertical kernel centered at pixel [i, j], so [i-2, j] to [i+2, j].
            for (int k = 0; k < 5; ++k) {
                const cv::Vec3i* tempRow = temp.ptr<cv::Vec3i>(i + k - 2);
                sum_b += kernel[k] * tempRow[j][0]; // Apply kernel to each channel.
                sum_g += kernel[k] * tempRow[j][1];
                sum_r += kernel[k] * tempRow[j][2];
            }

            dstRow[j][0] = static_cast<uchar>(sum_b / divisor); // Normalize and assign to output image.
            dstRow[j][1] = static_cast<uchar>(sum_g / divisor);
            dstRow[j][2] = static_cast<uchar>(sum_r / divisor);
        }
    }

    return 0;
}

int sobelX3x3(const cv::Mat &src, cv::Mat &dst ) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_16SC3); // Use signed type to hold negative values.
    dst.setTo(0); // Set initial values to zero for borders.

    std::vector<std::vector<int>> kernel;
    kernel = { {-1, 0, 1},
               {-2, 0, 2},
               {-1, 0, 1} };

    // Loop through each pixel in the image.
    for (int i = 1; i < src.rows - 1; ++i) {
        cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(i); // Use pointers for output.

        for (int j = 1; j < src.cols - 1; ++j) {
            int sum_b = 0, sum_g = 0, sum_r = 0;

            // Loop through the kernel centered at pixel [i, j] for all 3 RGB channels.
            for (int k = 0; k < 3; ++k) {
                const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i + k - 1); // Use pointers for input.
                
                for (int l = 0; l < 3; ++l) {
                    sum_b += kernel[k][l] * srcRow[j + l - 1][0];
                    sum_g += kernel[k][l] * srcRow[j + l - 1][1];
                    sum_r += kernel[k][l] * srcRow[j + l - 1][2];
                }
            }
            // Final values with sign.
            dstRow[j][0] = static_cast<short>(sum_b);
            dstRow[j][1] = static_cast<short>(sum_g);
            dstRow[j][2] = static_cast<short>(sum_r);
        }
    }

    return 0;
}   

int sobelY3x3(const cv::Mat &src, cv::Mat &dst ) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.size(), CV_16SC3); // Use signed type to hold negative values.
    dst.setTo(0); // Set initial values to zero for borders.

    std::vector<std::vector<int>> kernel;
    kernel = { {1, 2, 1},
               {0, 0, 0},
               {-1, -2, -1} };

    // Loop through each pixel in the image.
    for (int i = 1; i < src.rows - 1; ++i) {
        cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(i); // Use pointers for output.

        for (int j = 1; j < src.cols - 1; ++j) {
            int sum_b = 0, sum_g = 0, sum_r = 0;

            // Loop through the kernel centered at pixel [i, j] for all 3 RGB channels.
            for (int k = 0; k < 3; ++k) {
                const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i + k - 1); // Use pointers for input.
                
                for (int l = 0; l < 3; ++l) {
                    sum_b += kernel[k][l] * srcRow[j + l - 1][0];
                    sum_g += kernel[k][l] * srcRow[j + l - 1][1];
                    sum_r += kernel[k][l] * srcRow[j + l - 1][2];
                }
            }
            // Final values with sign.
            dstRow[j][0] = static_cast<short>(sum_b);
            dstRow[j][1] = static_cast<short>(sum_g);
            dstRow[j][2] = static_cast<short>(sum_r);
        }
    }

    return 0;
}

int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) {
        return -1;
    }

    dst.create(sx.size(), CV_8UC3); // Use 3 channel uchar type for final magnitude image, since sqrt give unsigned values.

    // Loop through each pixel in the image.
    for (int i = 0; i < sx.rows; ++i) {
        const cv::Vec3s* sxRow = sx.ptr<cv::Vec3s>(i); // Pointer to the sx row.
        const cv::Vec3s* syRow = sy.ptr<cv::Vec3s>(i); // Pointer to the sy row.
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i); // Pointer to the dst row. Type is 0-255 since sqrt gives positive only.

        for (int j = 0; j < sx.cols; ++j) {
            // Euclidean distance for each channel.
            for (int c = 0; c < 3; ++c) {
                int val_sx = static_cast<int>(sxRow[j][c]);
                int val_sy = static_cast<int>(syRow[j][c]);
                int magnitude = static_cast<int>(std::sqrt(val_sx * val_sx + val_sy * val_sy));
                dstRow[j][c] = static_cast<uchar>(magnitude); // uchar type for 0-255.
            }
        }
    }

    return 0;

}

int blurQuantize( const cv::Mat &src, cv::Mat &dst, int levels ) {
    if (src.empty() || levels <= 0) {
        return -1;
    }

    dst.create(src.size(), src.type()); // Initialize output image.

    cv::Mat tempBlur;
    blur5x5_2(src, tempBlur); // Apply blur to the input.

    int interval = 255 / levels; // Define quantization interval. It must be integer.

    // Loop through each pixel in the image.
    for (int i = 0; i < src.rows; ++i) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i); // Pointer to the input row.
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i); // Pointer to the output row.
        for (int j = 0; j < src.cols; ++j) {

            // Apply quantization to 3 channels.
            for (int c = 0; c < 3; ++c) {
                int quantized_value = (srcRow[j][c] / interval) * interval;
                if (quantized_value > 255) {
                    quantized_value = 255; // Ensure value does not exceed 255.
                }
                dstRow[j][c] = static_cast<uchar>(quantized_value);
            }
        }
    }
    return 0;
}
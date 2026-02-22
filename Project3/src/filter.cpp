/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 3.

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

int blur5x5_grey(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Ensure input is single-channel
    if (src.channels() != 1) {
        return -1;
    }

    src.copyTo(dst); // Initialize dst with src to keep border pixels.

    std::vector<int> kernel = {1, 2, 4, 2, 1};

    int sum_kernel = 0;
    for (int val : kernel) {
        sum_kernel += val;
    }
    int divisor = sum_kernel * sum_kernel;

    cv::Mat temp(src.rows, src.cols, CV_32SC1); // Single-channel int temp buffer.

    // Horizontal blur first.
    for (int i = 0; i < src.rows; ++i) {
        const uchar* srcRow = src.ptr<uchar>(i);     // Single-channel uchar pointer.
        int* tempRow = temp.ptr<int>(i);             // Single-channel int pointer.

        for (int j = 2; j < src.cols - 2; ++j) {
            int sum = 0;

            for (int k = 0; k < 5; ++k) {
                int idx = j + k - 2;
                sum += kernel[k] * srcRow[idx];      // Single channel — no [0],[1],[2].
            }

            tempRow[j] = sum;
        }
    }

    // Vertical blur.
    for (int i = 2; i < src.rows - 2; ++i) {
        uchar* dstRow = dst.ptr<uchar>(i);

        for (int j = 2; j < src.cols - 2; ++j) {
            int sum = 0;

            for (int k = 0; k < 5; ++k) {
                const int* tempRow = temp.ptr<int>(i + k - 2);
                sum += kernel[k] * tempRow[j];
            }

            dstRow[j] = static_cast<uchar>(sum / divisor);
        }
    }

    return 0;
}

// Converts a BGR image to a greyscale image representing the saturation channel.
int saturation(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
        uchar* dstRow = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; ++j) {
            // Normalize BGR channels to 0-1.
            float b = srcRow[j][0] / 255.0f;
            float g = srcRow[j][1] / 255.0f;
            float r = srcRow[j][2] / 255.0f;

            float cmax = std::max({r, g, b});
            float cmin = std::min({r, g, b});
            float delta = cmax - cmin;

            // Saturation: 0 if achromatic (cmax == 0), otherwise delta / cmax
            float s = (cmax == 0.0f) ? 0.0f : (delta / cmax);

            dstRow[j] = static_cast<uchar>(s * 255.0f);
        }
    }

    return 0;
}

int threshold(const cv::Mat &src, cv::Mat &dst, int thresh) {
    if (src.empty() || src.channels() != 1) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        const uchar* srcRow = src.ptr<uchar>(i);
        uchar* dstRow = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; ++j) {
            dstRow[j] = (srcRow[j] > thresh) ? 255 : 0;
        }
    }

    return 0;
}

// Compute threshold automatically using k-means clustering
int kmeansThreshold(const cv::Mat &src, int k, int maxIter) {
    // Histogram of pixel intensities 0-255
    int hist[256] = {0};
    for (int i = 0; i < src.rows; ++i) {
        const uchar* row = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
            hist[row[j]]++; // Populate histogram of intensity values at pixel[i, j]
    }

    // Initialize centroids evenly spaced across 0-255
    std::vector<float> centroids(k);
    for (int i = 0; i < k; ++i)
        centroids[i] = (i + 1) * 255.0f / (k + 1);

    for (int iter = 0; iter < maxIter; ++iter) {
        std::vector<float> sumVals(k, 0.0f);
        std::vector<int> counts(k, 0);

        // Assign each intensity value to nearest centroid
        for (int v = 0; v < 256; ++v) {
            if (hist[v] == 0) continue;

            int best = 0;
            float bestDist = std::abs(v - centroids[0]);
            for (int c = 1; c < k; ++c) {
                float dist = std::abs(v - centroids[c]);
                if (dist < bestDist) { bestDist = dist; best = c; }
            }
            sumVals[best] += v * hist[v];
            counts[best]  += hist[v];
        }

        // Recompute centroids, check for convergence
        bool converged = true;
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            float newCentroid = sumVals[c] / counts[c];
            if (std::abs(newCentroid - centroids[c]) > 0.5f) converged = false;
            centroids[c] = newCentroid;
        }
        if (converged) break;
    }

    // Sort centroids and return midpoint between the two lowest clusters
    std::sort(centroids.begin(), centroids.end());
    return static_cast<int>((centroids[0] + centroids[1]) / 2.0f);
}

int erode(const cv::Mat &src, cv::Mat &dst, int radius) {
    if (src.empty() || src.channels() != 1) return -1;
    dst.create(src.rows, src.cols, CV_8UC1);

    // Loop through each pixel
    for (int i = 0; i < src.rows; ++i) {
        uchar* dstRow = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j) {
            bool allWhite = true;
            // Loop through neighbours around the pixel[i,j]
            for (int di = -radius; di <= radius && allWhite; ++di)
                for (int dj = -radius; dj <= radius && allWhite; ++dj) {
                    int ni = i + di, nj = j + dj;
                    // The pixel[i,j] is while only if every neighbour is white. Out of image is considered black.
                    if (ni < 0 || ni >= src.rows || nj < 0 || nj >= src.cols)
                        allWhite = false;
                    else if (src.at<uchar>(ni, nj) == 0)
                        allWhite = false;
                }
            dstRow[j] = allWhite ? 255 : 0;
        }
    }
    return 0;
}

int dilate(const cv::Mat &src, cv::Mat &dst, int radius) {
    if (src.empty() || src.channels() != 1) return -1;
    dst.create(src.rows, src.cols, CV_8UC1);

    // Loop through each pixel
    for (int i = 0; i < src.rows; ++i) {
        uchar* dstRow = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j) {
            bool anyWhite = false;
            // Loop through neighbours around the pixel[i,j]
            for (int di = -radius; di <= radius && !anyWhite; ++di)
                for (int dj = -radius; dj <= radius && !anyWhite; ++dj) {
                    int ni = i + di, nj = j + dj;
                    // The pixel[i,j] is white only if any neighbour is white. Out of image is considered black.
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols)
                        if (src.at<uchar>(ni, nj) == 255)
                            anyWhite = true;
                }
            dstRow[j] = anyWhite ? 255 : 0;
        }
    }
    return 0;
}

int opening(const cv::Mat &src, cv::Mat &dst, int radius) {
    cv::Mat tmp;
    erode(src, tmp, radius);
    dilate(tmp, dst, radius);
    return 0;
}

int closing(const cv::Mat &src, cv::Mat &dst, int radius) {
    cv::Mat tmp;
    dilate(src, tmp, radius);
    erode(tmp, dst, radius);
    return 0;
}
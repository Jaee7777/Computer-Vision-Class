/*
Jaee Oh

CS5330 Computer Vision Project 1: Task 1

Referenced from OpenCV Tutorials (https://docs.opencv.org/4.5.1/db/deb/tutorial_display_image.html)
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;

// Function prototype
std::vector<float> extractFeatureVector(const cv::Mat &image);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    Mat image = imread(argv[1], IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Could not read the image!" << std::endl;
        return 1;
    }

    std::cout << "Feature Vector (center 7x7 pixels):" << std::endl;
    std::vector<float> feature_vector = extractFeatureVector(image);
    
    for (size_t i = 0; i < feature_vector.size(); ++i) {
        std::cout << feature_vector[i] << " ";
    }
    std::cout << std::endl;

    imshow("Display window", image);

    while (true) {
        int k = waitKey(0);

        if (k == 's') {
            imwrite("cat_copy.png", image);
        }
        else if (k == 'q') {
            break;
        }
    }

    return 0;
}

std::vector<float> extractFeatureVector(const cv::Mat &image) {
    std::vector<float> feature_vector;
    
    // Safety check
    if (image.cols < 7 || image.rows < 7) {
        std::cout << "Image too small for 7x7 feature extraction" << std::endl;
        return feature_vector;
    }
    
    // Pre-allocate
    feature_vector.reserve(7 * 7 * image.channels());
    
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;
    int start_x = center_x - 3;
    int start_y = center_y - 3;
    
    for (int row = start_y; row < start_y + 7; row++) {
        for (int col = start_x; col < start_x + 7; col++) {
            if (image.channels() == 3) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
                feature_vector.push_back(pixel[0]);
                feature_vector.push_back(pixel[1]);
                feature_vector.push_back(pixel[2]);
            } else {
                feature_vector.push_back(image.at<uchar>(row, col));
            }
        }
    }
    
    return feature_vector;
}

float compute_ssd(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
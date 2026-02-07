/*
  Jaee Oh
  Project 2

  Modified from the template given by Professor Maxwell
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <functional>

// Function declarations
std::vector<float> extract_feature_vector(const cv::Mat &image);
float compute_ssd(const std::vector<float> &a, const std::vector<float> &b);
std::vector<std::vector<std::vector<float>>> compute_3D_histogram(const cv::Mat &image, int bins);
std::vector<std::vector<std::vector<float>>> compute_3D_histogram_region(const cv::Mat &image, int bins, int start_row, int end_row);
float histogram_intersection_3D(const std::vector<std::vector<std::vector<float>>> &hist1, 
                                const std::vector<std::vector<std::vector<float>>> &hist2);

struct ImageMatch {
    std::string filename;
    float score;
};

bool compare_ascending(const ImageMatch &a, const ImageMatch &b) {
    return a.score < b.score;
}

bool compare_descending(const ImageMatch &a, const ImageMatch &b) {
    return a.score > b.score;
}

/*
 * Main function
 */
int main(int argc, char *argv[]) {
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;

    // Check for sufficient arguments
    if(argc < 5) {
        printf("usage: %s <directory path> <target file path> [Top N matches] [Matching method]\n", argv[0]);
        exit(-1);
    }

    // Get the directory path
    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);

    // Open the directory
    dirp = opendir(dirname);
    if(dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // Read target image
    cv::Mat target_image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if(target_image.empty()) {
        printf("Could not read target image %s\n", argv[2]);
        closedir(dirp);
        exit(-1);
    }
    
    // Parse top N with error handling
    int top_n;
    try {
        top_n = std::stoi(argv[3]);
        if(top_n <= 0) {
            printf("Top N must be positive\n");
            closedir(dirp);
            exit(-1);
        }
    } catch(const std::exception& e) {
        printf("Invalid number for Top N: %s\n", argv[3]);
        closedir(dirp);
        exit(-1);
    }
    
    // Check matching method
    std::string method = argv[4];

    if(method == "ssd") {
        printf("Using SSD matching method\n");
        
        // Find feature vector for the target image
        std::vector<float> target_feature_vector = extract_feature_vector(target_image);

        // Vector to store file names and scores
        std::vector<ImageMatch> matches;

        // Loop over all the files in the image file listing
        while((dp = readdir(dirp)) != NULL) {

            // Check if the file is an image
            if(strstr(dp->d_name, ".jpg") ||
               strstr(dp->d_name, ".png") ||
               strstr(dp->d_name, ".ppm") ||
               strstr(dp->d_name, ".tif")) {

                printf("processing image file: %s\n", dp->d_name);

                // Build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                printf("full path name: %s\n", buffer);

                // Find feature vector for the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                std::vector<float> feature_vector = extract_feature_vector(image);

                // Find SSD
                float ssd = compute_ssd(target_feature_vector, feature_vector);
                printf("SSD of image with target: %.2f\n", ssd);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = ssd;
                matches.push_back(match);
            }
        }

        // Sort matches by SSD (ascending - lower is better)
        std::sort(matches.begin(), matches.end(), compare_ascending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with SSD = %.2f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }

    } else if(method == "histogram") {
        printf("Using histogram intersection matching method (RGB color space, 8x8x8 bins)\n");
        
        // Find normalized 3D histogram for the target image (8 bins per channel)
        std::vector<std::vector<std::vector<float>>> target_histogram = compute_3D_histogram(target_image, 8);

        // Vector to store file names and scores
        std::vector<ImageMatch> matches;

        // Loop over all the files in the image file listing
        while((dp = readdir(dirp)) != NULL) {

            // Check if the file is an image
            if(strstr(dp->d_name, ".jpg") ||
               strstr(dp->d_name, ".png") ||
               strstr(dp->d_name, ".ppm") ||
               strstr(dp->d_name, ".tif")) {

                printf("processing image file: %s\n", dp->d_name);

                // Build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                printf("full path name: %s\n", buffer);

                // Find histogram for the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                std::vector<std::vector<std::vector<float>>> histogram = compute_3D_histogram(image, 8);

                // Find histogram intersection
                float intersection = histogram_intersection_3D(target_histogram, histogram);
                printf("Histogram intersection of image with target: %.4f\n", intersection);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = intersection;
                matches.push_back(match);
            }
        }

        // Sort matches by intersection (descending - higher is better)
        std::sort(matches.begin(), matches.end(), compare_descending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with intersection = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "multi-histogram") {
        printf("Using multi histogram intersection matching method (RGB color space, 8x8x8 bins, top/bottom regions)\n");
        
        // Validate image has sufficient height
        if(target_image.rows < 2) {
            printf("Target image too small for multi-histogram method\n");
            closedir(dirp);
            exit(-1);
        }
        
        // Find normalized 3D histograms for top and bottom regions of target image
        std::vector<std::vector<std::vector<float>>> target_histogram_top = 
            compute_3D_histogram_region(target_image, 8, 0, target_image.rows / 2);
        std::vector<std::vector<std::vector<float>>> target_histogram_bottom = 
            compute_3D_histogram_region(target_image, 8, target_image.rows / 2, target_image.rows);
        
        // Vector to store file names and scores
        std::vector<ImageMatch> matches;

        // Loop over all the files in the image file listing
        while((dp = readdir(dirp)) != NULL) {

            // Check if the file is an image
            if(strstr(dp->d_name, ".jpg") ||
               strstr(dp->d_name, ".png") ||
               strstr(dp->d_name, ".ppm") ||
               strstr(dp->d_name, ".tif")) {

                printf("processing image file: %s\n", dp->d_name);

                // Build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                printf("full path name: %s\n", buffer);

                // Find histogram for the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                // Check image has sufficient height
                if(image.rows < 2) {
                    printf("Image too small for multi-histogram: %s\n", buffer);
                    continue;
                }
                
                std::vector<std::vector<std::vector<float>>> histogram_top = 
                    compute_3D_histogram_region(image, 8, 0, image.rows / 2);
                std::vector<std::vector<std::vector<float>>> histogram_bottom = 
                    compute_3D_histogram_region(image, 8, image.rows / 2, image.rows);

                // Find histogram intersection (averaged to keep score in 0-1 range)
                float intersection_top = histogram_intersection_3D(target_histogram_top, histogram_top);
                float intersection_bottom = histogram_intersection_3D(target_histogram_bottom, histogram_bottom);
                float intersection = (intersection_top + intersection_bottom) / 2.0f;
                
                printf("Histogram intersection of image with target: %.4f (top: %.4f, bottom: %.4f)\n", 
                       intersection, intersection_top, intersection_bottom);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = intersection;
                matches.push_back(match);
            }
        }

        // Sort matches by intersection (descending - higher is better)
        std::sort(matches.begin(), matches.end(), compare_descending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with intersection = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else {
        printf("Unknown matching method: %s\n", method.c_str());
        printf("Available methods: ssd, histogram, multi-histogram\n");
        closedir(dirp);
        exit(-1);
    }

    printf("\nTerminating\n");
    closedir(dirp);  
    return 0;
}

// Extract 7x7 feature vector from center of image
std::vector<float> extract_feature_vector(const cv::Mat &image) {
    std::vector<float> feature_vector;
    
    // Check size of image
    if(image.cols < 7 || image.rows < 7) {
        std::cout << "Image too small for 7x7 feature extraction" << std::endl;
        return feature_vector;
    }
    
    // Pre-allocate
    feature_vector.reserve(7 * 7 * image.channels());
    
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;
    int start_x = center_x - 3;
    int start_y = center_y - 3;
    
    for(int row = start_y; row < start_y + 7; row++) {
        for(int col = start_x; col < start_x + 7; col++) {
            if(image.channels() == 3) {
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

// Compute Sum of Squared Differences
float compute_ssd(const std::vector<float> &a, const std::vector<float> &b) {
    // Safety check
    if(a.size() != b.size()) {
        printf("Error: Feature vectors have different sizes\n");
        return -1.0f;
    }
    
    float sum = 0.0f;
    for(size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Compute 3D histogram (RGB color space with 8 bins per channel)
std::vector<std::vector<std::vector<float>>> compute_3D_histogram(const cv::Mat &image, int bins) {
    // Initialize 3D histogram with zeros (bins x bins x bins)
    std::vector<std::vector<std::vector<float>>> histogram(
        bins, 
        std::vector<std::vector<float>>(
            bins, 
            std::vector<float>(bins, 0.0f)
        )
    );
    
    // Compute histogram for entire image
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            
            // OpenCV uses BGR format: pixel[2]=Red, pixel[1]=Green, pixel[0]=Blue
            int r_bin = (pixel[2] * bins) / 256;
            int g_bin = (pixel[1] * bins) / 256;
            int b_bin = (pixel[0] * bins) / 256;
            
            // Clamp to valid range
            r_bin = std::min(r_bin, bins - 1);
            g_bin = std::min(g_bin, bins - 1);
            b_bin = std::min(b_bin, bins - 1);
            
            // Increment the 3D bin
            histogram[r_bin][g_bin][b_bin]++;
        }
    }

    // Normalize histogram
    float sum = 0.0f;
    for(const auto &plane : histogram) {
        for(const auto &row : plane) {
            for(float value : row) {
                sum += value;
            }
        }
    }

    // Normalize if sum is greater than 0
    if(sum > 0) {
        for(auto &plane : histogram) {
            for(auto &row : plane) {
                for(float &value : row) {
                    value /= sum;
                }
            }
        }
    }
    
    return histogram;
}

// Compute 3D histogram for a specific row region (RGB color space with 8 bins per channel)
std::vector<std::vector<std::vector<float>>> compute_3D_histogram_region(
    const cv::Mat &image, int bins, int start_row, int end_row) {
    
    // Initialize 3D histogram with zeros (bins x bins x bins)
    std::vector<std::vector<std::vector<float>>> histogram(
        bins, 
        std::vector<std::vector<float>>(
            bins, 
            std::vector<float>(bins, 0.0f)
        )
    );
    
    // Compute histogram for specified row range
    for(int i = start_row; i < end_row; i++) {
        for(int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            
            // OpenCV uses BGR format: pixel[2]=Red, pixel[1]=Green, pixel[0]=Blue
            int r_bin = (pixel[2] * bins) / 256;
            int g_bin = (pixel[1] * bins) / 256;
            int b_bin = (pixel[0] * bins) / 256;
            
            // Clamp to valid range
            r_bin = std::min(r_bin, bins - 1);
            g_bin = std::min(g_bin, bins - 1);
            b_bin = std::min(b_bin, bins - 1);
            
            // Increment the 3D bin
            histogram[r_bin][g_bin][b_bin]++;
        }
    }

    // Normalize histogram
    float sum = 0.0f;
    for(const auto &plane : histogram) {
        for(const auto &row : plane) {
            for(float value : row) {
                sum += value;
            }
        }
    }

    // Normalize if sum is greater than 0
    if(sum > 0) {
        for(auto &plane : histogram) {
            for(auto &row : plane) {
                for(float &value : row) {
                    value /= sum;
                }
            }
        }
    }
    
    return histogram;
}

// Compute histogram intersection for 3D histograms
float histogram_intersection_3D(const std::vector<std::vector<std::vector<float>>> &hist1, 
                                const std::vector<std::vector<std::vector<float>>> &hist2) {
    // Check dimensions match
    if(hist1.size() != hist2.size() || 
       hist1[0].size() != hist2[0].size() || 
       hist1[0][0].size() != hist2[0][0].size()) {
        printf("Error: Histogram dimensions don't match\n");
        return 0.0f;
    }
    
    float intersection = 0.0f;
    for(size_t i = 0; i < hist1.size(); i++) {
        for(size_t j = 0; j < hist1[i].size(); j++) {
            for(size_t k = 0; k < hist1[i][j].size(); k++) {
                intersection += std::min(hist1[i][j][k], hist2[i][j][k]);
            }
        }
    }
    return intersection;
}
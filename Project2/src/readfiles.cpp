/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 2.

  Modified from the template given by Professor Maxwell
  
  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Autocorrection done by local Qwen 2.5 model.
  Claude AI was used for code review and debugging.

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
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>

// Function declarations
std::vector<float> extract_feature_vector(const cv::Mat &image);
float compute_ssd(const std::vector<float> &a, const std::vector<float> &b);
std::vector<std::vector<std::vector<float>>> compute_3D_histogram(const cv::Mat &image, int bins);
std::vector<std::vector<std::vector<float>>> compute_3D_histogram_region(const cv::Mat &image, int bins, int start_row, int end_row);
std::vector<std::vector<float>> compute_texture_histogram(const cv::Mat &image, int mag_bins, int angle_bins);
std::vector<float> compute_gabor_features(const cv::Mat &image);
cv::Mat create_gabor_kernel(int ksize, double sigma, double theta, double lambda, double gamma, double psi);
float histogram_intersection_3D(const std::vector<std::vector<std::vector<float>>> &hist1, 
                                const std::vector<std::vector<std::vector<float>>> &hist2);
float histogram_intersection_2D(const std::vector<std::vector<float>> &hist1,
                                const std::vector<std::vector<float>> &hist2);
float compute_cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);
float compute_cosine_distance(const std::vector<float> &a, const std::vector<float> &b);
std::map<std::string, std::vector<float>> read_features_from_csv(const std::string &csv_path);

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
        printf("usage: %s <directory path> <target file path> [Top N matches] [Matching method] [optional: csv file path]\n", argv[0]);
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

    if(method == "csv-cosine") {
        // Check if CSV file path is provided
        if(argc < 6) {
            printf("Error: CSV file path required for csv-cosine method\n");
            printf("usage: %s <directory path> <target file path> [Top N matches] csv-cosine <csv file path>\n", argv[0]);
            closedir(dirp);
            exit(-1);
        }
        
        std::string csv_path = argv[5];
        printf("Using CSV-based cosine distance matching method\n");
        printf("Reading features from: %s\n", csv_path.c_str());
        
        // Read all features from CSV
        std::map<std::string, std::vector<float>> feature_database = read_features_from_csv(csv_path);
        
        if(feature_database.empty()) {
            printf("Error: Could not read features from CSV file or CSV is empty\n");
            closedir(dirp);
            exit(-1);
        }
        
        printf("Loaded %zu feature vectors from CSV\n", feature_database.size());
        
        // Get the target filename (basename without path)
        std::string target_path = argv[2];
        size_t last_slash = target_path.find_last_of("/\\");
        std::string target_filename = (last_slash != std::string::npos) ? 
                                      target_path.substr(last_slash + 1) : target_path;
        
        // Find target features in database
        auto target_it = feature_database.find(target_filename);
        if(target_it == feature_database.end()) {
            printf("Error: Target image '%s' not found in CSV feature database\n", target_filename.c_str());
            closedir(dirp);
            exit(-1);
        }
        
        std::vector<float> target_features = target_it->second;
        printf("Target image: %s (feature dimension: %zu)\n", target_filename.c_str(), target_features.size());
        
        // Vector to store matches
        std::vector<ImageMatch> matches;
        
        // Compare with all images in the database
        for(const auto &entry : feature_database) {
            const std::string &filename = entry.first;
            const std::vector<float> &features = entry.second;
            
            // Skip the target image itself
            if(filename == target_filename) {
                continue;
            }
            
            // Compute cosine distance
            float distance = compute_cosine_distance(target_features, features);
            
            printf("Comparing with %s: cosine distance = %.4f\n", filename.c_str(), distance);
            
            // Store the result
            ImageMatch match;
            match.filename = filename;
            match.score = distance;
            matches.push_back(match);
        }
        
        // Sort by distance (ascending - lower distance is better)
        std::sort(matches.begin(), matches.end(), compare_ascending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with cosine distance = %.4f\n", 
                   i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "ssd") {
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
        
    } else if(method == "color-texture") {
        printf("Using color + texture histogram matching method\n");
        printf("Color: RGB 8x8x8 bins, Texture: Gradient magnitude & orientation 16x16 bins\n");
        
        // Compute target histograms
        std::vector<std::vector<std::vector<float>>> target_color_hist = compute_3D_histogram(target_image, 8);
        std::vector<std::vector<float>> target_texture_hist = compute_texture_histogram(target_image, 16, 16);
        
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

                // Find histograms for the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                std::vector<std::vector<std::vector<float>>> color_hist = compute_3D_histogram(image, 8);
                std::vector<std::vector<float>> texture_hist = compute_texture_histogram(image, 16, 16);

                // Compute intersections
                float color_intersection = histogram_intersection_3D(target_color_hist, color_hist);
                float texture_intersection = histogram_intersection_2D(target_texture_hist, texture_hist);
                
                // Weighted combination (50% color, 50% texture)
                float combined_score = 0.5f * color_intersection + 0.5f * texture_intersection;
                
                printf("Color: %.4f, Texture: %.4f, Combined: %.4f\n", 
                       color_intersection, texture_intersection, combined_score);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = combined_score;
                matches.push_back(match);
            }
        }

        // Sort matches by combined score (descending - higher is better)
        std::sort(matches.begin(), matches.end(), compare_descending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with combined score = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "gabor") {
        printf("Using Gabor filter-based texture matching method\n");
        printf("Extracting multi-scale, multi-orientation Gabor features\n");
        
        // Compute target Gabor features
        std::vector<float> target_gabor_features = compute_gabor_features(target_image);
        
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

                // Find Gabor features for the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                std::vector<float> gabor_features = compute_gabor_features(image);

                // Compute similarity using cosine similarity
                float similarity = compute_cosine_similarity(target_gabor_features, gabor_features);
                
                printf("Gabor similarity: %.4f\n", similarity);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = similarity;
                matches.push_back(match);
            }
        }

        // Sort matches by similarity (descending - higher is better)
        std::sort(matches.begin(), matches.end(), compare_descending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with Gabor similarity = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "color-gabor") {
        printf("Using color + Gabor texture matching method\n");
        printf("Color: RGB 8x8x8 bins, Texture: Multi-scale Gabor filters\n");
        
        // Compute target features
        std::vector<std::vector<std::vector<float>>> target_color_hist = compute_3D_histogram(target_image, 8);
        std::vector<float> target_gabor_features = compute_gabor_features(target_image);
        
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

                // Find features for the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                std::vector<std::vector<std::vector<float>>> color_hist = compute_3D_histogram(image, 8);
                std::vector<float> gabor_features = compute_gabor_features(image);

                // Compute similarities
                float color_intersection = histogram_intersection_3D(target_color_hist, color_hist);
                float gabor_similarity = compute_cosine_similarity(target_gabor_features, gabor_features);
                
                // Weighted combination (50% color, 50% Gabor texture)
                float combined_score = 0.5f * color_intersection + 0.5f * gabor_similarity;
                
                printf("Color: %.4f, Gabor: %.4f, Combined: %.4f\n", 
                       color_intersection, gabor_similarity, combined_score);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = combined_score;
                matches.push_back(match);
            }
        }

        // Sort matches by combined score (descending - higher is better)
        std::sort(matches.begin(), matches.end(), compare_descending);
        
        printf("\nTop %d matches:\n", top_n);
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with combined score = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else {
        printf("Unknown matching method: %s\n", method.c_str());
        printf("Available methods: ssd, histogram, multi-histogram, color-texture, gabor, color-gabor, csv-cosine, csv-cosine-similarity\n");
        closedir(dirp);
        exit(-1);
    }

    printf("\nTerminating\n");
    closedir(dirp);  
    return 0;
}

// Read feature vectors from CSV file
// Expected format: filename, feature1, feature2, ..., feature512
std::map<std::string, std::vector<float>> read_features_from_csv(const std::string &csv_path) {
    std::map<std::string, std::vector<float>> feature_database;
    
    std::ifstream file(csv_path);
    if(!file.is_open()) {
        printf("Error: Could not open CSV file: %s\n", csv_path.c_str());
        return feature_database;
    }
    
    std::string line;
    int line_number = 0;
    
    // Read header line (optional, skip if exists)
    if(std::getline(file, line)) {
        line_number++;
        // Check if first line is a header by seeing if first column contains "filename" or similar
        if(line.find("filename") != std::string::npos || line.find("image") != std::string::npos) {
            printf("Skipping header line\n");
        } else {
            // First line is data, process it
            file.seekg(0);  // Reset to beginning
            line_number = 0;
        }
    }
    
    // Read each line
    while(std::getline(file, line)) {
        line_number++;
        
        if(line.empty()) continue;
        
        std::stringstream ss(line);
        std::string filename;
        std::vector<float> features;
        
        // Read filename (first column)
        if(!std::getline(ss, filename, ',')) {
            printf("Warning: Line %d - Could not read filename\n", line_number);
            continue;
        }
        
        // Trim whitespace from filename
        filename.erase(0, filename.find_first_not_of(" \t\r\n"));
        filename.erase(filename.find_last_not_of(" \t\r\n") + 1);
        
        // Read feature values
        std::string value_str;
        while(std::getline(ss, value_str, ',')) {
            try {
                float value = std::stof(value_str);
                features.push_back(value);
            } catch(const std::exception& e) {
                printf("Warning: Line %d - Could not parse feature value: %s\n", 
                       line_number, value_str.c_str());
                continue;
            }
        }
        
        if(features.empty()) {
            printf("Warning: Line %d - No features found for %s\n", line_number, filename.c_str());
            continue;
        }
        
        // Store in database
        feature_database[filename] = features;
        printf("Loaded %s with %zu features\n", filename.c_str(), features.size());
    }
    
    file.close();
    printf("Successfully loaded %zu entries from CSV\n", feature_database.size());
    
    return feature_database;
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

// Compute 2D texture histogram based on gradient magnitude and orientation
std::vector<std::vector<float>> compute_texture_histogram(const cv::Mat &image, int mag_bins, int angle_bins) {
    // Initialize 2D histogram with zeros
    std::vector<std::vector<float>> histogram(mag_bins, std::vector<float>(angle_bins, 0.0f));
    
    // Convert to grayscale
    cv::Mat gray;
    if(image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Compute gradients using Sobel operator
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);  // Gradient in x direction
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);  // Gradient in y direction
    
    // Compute magnitude and angle for each pixel
    for(int i = 0; i < gray.rows; i++) {
        for(int j = 0; j < gray.cols; j++) {
            float gx = grad_x.at<float>(i, j);
            float gy = grad_y.at<float>(i, j);
            
            // Calculate magnitude
            float magnitude = std::sqrt(gx * gx + gy * gy);
            
            // Calculate angle in degrees (0-360)
            float angle = std::atan2(gy, gx) * 180.0f / M_PI;
            if(angle < 0) angle += 360.0f;
            
            // Determine bin indices
            // Magnitude bins: divide by reasonable max value (e.g., 255 for 8-bit)
            int mag_bin = static_cast<int>((magnitude * mag_bins) / 256.0f);
            mag_bin = std::min(mag_bin, mag_bins - 1);
            
            // Angle bins: 0-360 degrees
            int angle_bin = static_cast<int>((angle * angle_bins) / 360.0f);
            angle_bin = std::min(angle_bin, angle_bins - 1);
            
            // Increment histogram
            histogram[mag_bin][angle_bin]++;
        }
    }
    
    // Normalize histogram
    float sum = 0.0f;
    for(const auto &row : histogram) {
        for(float value : row) {
            sum += value;
        }
    }
    
    if(sum > 0) {
        for(auto &row : histogram) {
            for(float &value : row) {
                value /= sum;
            }
        }
    }
    
    return histogram;
}

// Create a Gabor kernel
cv::Mat create_gabor_kernel(int ksize, double sigma, double theta, double lambda, double gamma, double psi) {
    cv::Mat kernel(ksize, ksize, CV_32F);
    int half_size = ksize / 2;
    
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;
    
    for(int y = -half_size; y <= half_size; y++) {
        for(int x = -half_size; x <= half_size; x++) {
            // Rotate coordinates
            double x_theta = x * std::cos(theta) + y * std::sin(theta);
            double y_theta = -x * std::sin(theta) + y * std::cos(theta);
            
            // Gabor formula
            double gaussian = std::exp(-0.5 * (x_theta * x_theta / (sigma_x * sigma_x) + 
                                                y_theta * y_theta / (sigma_y * sigma_y)));
            double sinusoid = std::cos(2.0 * M_PI * x_theta / lambda + psi);
            
            kernel.at<float>(y + half_size, x + half_size) = gaussian * sinusoid;
        }
    }
    
    return kernel;
}

// Compute Gabor features for an image
std::vector<float> compute_gabor_features(const cv::Mat &image) {
    std::vector<float> features;
    
    // Convert to grayscale
    cv::Mat gray;
    if(image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Convert to float
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F);
    
    // Gabor filter parameters
    int ksize = 21;                           // Kernel size
    double sigma = 5.0;                       // Standard deviation
    double gamma = 0.5;                       // Aspect ratio
    double psi = 0;                           // Phase offset
    
    // Multiple scales (wavelengths)
    std::vector<double> lambdas = {5.0, 10.0, 15.0, 20.0};  // 4 scales
    
    // Multiple orientations
    std::vector<double> thetas = {0, M_PI/4, M_PI/2, 3*M_PI/4};  // 4 orientations (0, 45, 90, 135 degrees)
    
    // Apply Gabor filters at different scales and orientations
    for(double lambda : lambdas) {
        for(double theta : thetas) {
            // Create Gabor kernel
            cv::Mat kernel = create_gabor_kernel(ksize, sigma, theta, lambda, gamma, psi);
            
            // Apply filter
            cv::Mat filtered;
            cv::filter2D(gray_float, filtered, CV_32F, kernel);
            
            // Compute statistics (mean and standard deviation) of the filtered response
            cv::Scalar mean, stddev;
            cv::meanStdDev(filtered, mean, stddev);
            
            // Use mean and stddev as features
            features.push_back(mean[0]);
            features.push_back(stddev[0]);
        }
    }
    
    // Normalize features to [0, 1] range
    if(!features.empty()) {
        float max_val = *std::max_element(features.begin(), features.end());
        float min_val = *std::min_element(features.begin(), features.end());
        float range = max_val - min_val;
        
        if(range > 0) {
            for(float &f : features) {
                f = (f - min_val) / range;
            }
        }
    }
    
    return features;
}

// Compute cosine similarity between two feature vectors (returns 0 to 1, higher is better)
float compute_cosine_similarity(const std::vector<float> &a, const std::vector<float> &b) {
    if(a.size() != b.size() || a.empty()) {
        printf("Error: Feature vectors have incompatible sizes for cosine similarity\n");
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for(size_t i = 0; i < a.size(); i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if(norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

// Compute cosine distance between two feature vectors (returns 0 to 2, lower is better)
float compute_cosine_distance(const std::vector<float> &a, const std::vector<float> &b) {
    float similarity = compute_cosine_similarity(a, b);
    return 1.0f - similarity;  // Distance = 1 - similarity
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

// Compute histogram intersection for 2D histograms
float histogram_intersection_2D(const std::vector<std::vector<float>> &hist1,
                                const std::vector<std::vector<float>> &hist2) {
    // Check dimensions match
    if(hist1.size() != hist2.size() || hist1[0].size() != hist2[0].size()) {
        printf("Error: 2D Histogram dimensions don't match\n");
        return 0.0f;
    }
    
    float intersection = 0.0f;
    for(size_t i = 0; i < hist1.size(); i++) {
        for(size_t j = 0; j < hist1[i].size(); j++) {
            intersection += std::min(hist1[i][j], hist2[i][j]);
        }
    }
    return intersection;
}
/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 2.

  Modified from the templates given by Professor Maxwell
  
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
#include "opencv2/opencv.hpp"  // the top include file
#include "opencv2/dnn.hpp"     // DNN API include file
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <functional>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <sys/time.h>
#include <cmath>
#include "../include/faceDetect.h"

// =======================================================================
// Function prototypes and data structures
// =======================================================================
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
std::string get_basename(const std::string &path);
std::vector<cv::Vec3f> extract_dominant_colors_kmeans(const cv::Mat &image, int k = 5);
std::vector<float> compute_edge_density_map(const cv::Mat &image, int grid_size = 4);
float compute_chi_square_distance_3D(const std::vector<std::vector<std::vector<float>>> &hist1,
                                     const std::vector<std::vector<std::vector<float>>> &hist2);
float compute_chi_square_distance_2D(const std::vector<std::vector<float>> &hist1,
                                     const std::vector<std::vector<float>> &hist2);
float compute_chi_square_distance_1D(const std::vector<float> &hist1, const std::vector<float> &hist2);
float compute_hamming_distance(const std::vector<float> &a, const std::vector<float> &b);

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
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug=0 );

// =======================================================================
// Main function
// =======================================================================
int main(int argc, char *argv[]) {
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;

    // Check for sufficient arguments
    if(argc < 5) {
        printf("usage: %s <directory path> <target file path> [Top N matches] [Matching method] [optional: csv file path] [optional: --least-similar]\n", argv[0]);
        printf("Examples:\n");
        printf("  %s images/ target.jpg 5 histogram\n", argv[0]);
        printf("  %s images/ target.jpg 5 dominant-color-texture\n", argv[0]);
        printf("  %s images/ target.jpg 5 edge-density\n", argv[0]);
        printf("  %s images/ target.jpg 5 chi-square\n", argv[0]);
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
    
    // Check for --least-similar flag
    bool show_least_similar = false;
    for(int i = 5; i < argc; i++) {
        if(std::string(argv[i]) == "--least-similar") {
            show_least_similar = true;
            printf("*** Showing LEAST similar matches ***\n");
            break;
        }
    }

    if(method == "dominant-color-texture") {
        printf("Using Dominant Color + Texture matching method\n");
        printf("Extracting 5 dominant colors via K-means clustering\n");
        printf("Texture: Gradient magnitude & orientation 16x16 bins\n");
        
        // Extract target features
        std::vector<cv::Vec3f> target_dominant_colors = extract_dominant_colors_kmeans(target_image, 5);
        std::vector<std::vector<float>> target_texture_hist = compute_texture_histogram(target_image, 16, 16);
        
        // Vector to store matches
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

                // Read the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                // Extract features
                std::vector<cv::Vec3f> dominant_colors = extract_dominant_colors_kmeans(image, 5);
                std::vector<std::vector<float>> texture_hist = compute_texture_histogram(image, 16, 16);
                
                // Compare dominant colors (average minimum distance between color sets)
                float color_distance = 0.0f;
                for(const auto& target_color : target_dominant_colors) {
                    float min_dist = std::numeric_limits<float>::max();
                    for(const auto& img_color : dominant_colors) {
                        float dist = cv::norm(target_color - img_color);
                        min_dist = std::min(min_dist, dist);
                    }
                    color_distance += min_dist;
                }
                color_distance /= target_dominant_colors.size();
                
                // Normalize color distance to 0-1 range (assume max distance ~441 for RGB)
                float color_similarity = 1.0f - std::min(color_distance / 441.0f, 1.0f);
                
                // Texture similarity
                float texture_similarity = histogram_intersection_2D(target_texture_hist, texture_hist);
                
                // Combined score (50% color, 50% texture)
                float combined_score = 0.5f * color_similarity + 0.5f * texture_similarity;
                
                printf("  Color: %.4f, Texture: %.4f, Combined: %.4f\n",
                       color_similarity, texture_similarity, combined_score);
                
                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = combined_score;
                matches.push_back(match);
            }
        }
        
        // Sort matches
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with score = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "edge-density") {
        printf("Using Edge Density Map matching method\n");
        printf("Computing edge density in 4x4 grid\n");
        
        // Compute target edge density map
        std::vector<float> target_edge_density = compute_edge_density_map(target_image, 4);
        
        // Vector to store matches
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

                // Read the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                // Compute edge density map
                std::vector<float> edge_density = compute_edge_density_map(image, 4);
                
                // Compute chi-square distance
                float chi_square = compute_chi_square_distance_1D(target_edge_density, edge_density);
                
                // Convert to similarity (lower chi-square = more similar)
                float similarity = 1.0f / (1.0f + chi_square);
                
                printf("  Chi-square distance: %.4f, Similarity: %.4f\n", chi_square, similarity);
                
                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = similarity;
                matches.push_back(match);
            }
        }
        
        // Sort matches
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with similarity = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "chi-square") {
        printf("Using Chi-Square distance with RGB histogram matching method\n");
        
        // Compute target histogram
        std::vector<std::vector<std::vector<float>>> target_histogram = compute_3D_histogram(target_image, 8);
        
        // Vector to store matches
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

                // Read the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                // Compute histogram
                std::vector<std::vector<std::vector<float>>> histogram = compute_3D_histogram(image, 8);
                
                // Compute chi-square distance
                float chi_square = compute_chi_square_distance_3D(target_histogram, histogram);
                
                // Convert to similarity (lower chi-square = more similar)
                float similarity = 1.0f / (1.0f + chi_square);
                
                printf("  Chi-square distance: %.4f, Similarity: %.4f\n", chi_square, similarity);
                
                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = similarity;
                matches.push_back(match);
            }
        }
        
        // Sort matches
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with similarity = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "hamming") {
        printf("Using Hamming distance with binarized feature vectors\n");
        
        // Compute target feature vector and binarize
        std::vector<float> target_features = extract_feature_vector(target_image);
        
        // Vector to store matches
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

                // Read the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("Could not read image %s\n", buffer);
                    continue;
                }
                
                // Compute feature vector
                std::vector<float> features = extract_feature_vector(image);
                
                // Compute Hamming distance
                float hamming = compute_hamming_distance(target_features, features);
                
                // Convert to similarity
                float similarity = 1.0f - (hamming / target_features.size());
                
                printf("  Hamming distance: %.0f, Similarity: %.4f\n", hamming, similarity);
                
                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = similarity;
                matches.push_back(match);
            }
        }
        
        // Sort matches
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with similarity = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "dnn-color-texture") {
        // Check if CSV file path is provided
        if(argc < 6 || std::string(argv[5]) == "--least-similar") {
            printf("Error: CSV file path required for dnn-color-texture method\n");
            printf("usage: %s <directory path> <target file path> [Top N matches] dnn-color-texture <csv file path> [--least-similar]\n", argv[0]);
            closedir(dirp);
            exit(-1);
        }
        
        std::string csv_path = argv[5];
        printf("=== Hybrid DNN + Color + Texture Matching Method ===\n");
        printf("Components:\n");
        printf("  1. DNN features (from CSV) - 33.3%% weight\n");
        printf("  2. Color histogram (RGB 8x8x8) - 33.3%% weight\n");
        printf("  3. Texture histogram (Gradient mag/angle 16x16) - 33.3%% weight\n");
        printf("Reading DNN features from: %s\n", csv_path.c_str());
        
        // Read all DNN features from CSV
        std::map<std::string, std::vector<float>> dnn_feature_database = read_features_from_csv(csv_path);
        
        if(dnn_feature_database.empty()) {
            printf("Error: Could not read features from CSV file or CSV is empty\n");
            closedir(dirp);
            exit(-1);
        }
        
        printf("Loaded %zu DNN feature vectors from CSV\n", dnn_feature_database.size());
        
        // Get the target filename (basename without path)
        std::string target_path = argv[2];
        std::string target_filename = get_basename(target_path);
        
        // Find target DNN features in database
        auto target_dnn_it = dnn_feature_database.find(target_filename);
        if(target_dnn_it == dnn_feature_database.end()) {
            printf("Error: Target image '%s' not found in CSV feature database\n", target_filename.c_str());
            closedir(dirp);
            exit(-1);
        }
        
        std::vector<float> target_dnn_features = target_dnn_it->second;
        printf("Target image: %s (DNN feature dimension: %zu)\n", target_filename.c_str(), target_dnn_features.size());
        
        // Compute target color and texture features
        printf("Computing target color histogram...\n");
        std::vector<std::vector<std::vector<float>>> target_color_hist = compute_3D_histogram(target_image, 8);
        
        printf("Computing target texture histogram...\n");
        std::vector<std::vector<float>> target_texture_hist = compute_texture_histogram(target_image, 16, 16);
        
        // Vector to store matches
        std::vector<ImageMatch> matches;
        
        // Loop over all the files in the image directory
        while((dp = readdir(dirp)) != NULL) {
            // Check if the file is an image
            if(strstr(dp->d_name, ".jpg") ||
               strstr(dp->d_name, ".png") ||
               strstr(dp->d_name, ".ppm") ||
               strstr(dp->d_name, ".tif")) {

                std::string current_filename = dp->d_name;
                
                // Skip the target image itself
                if(current_filename == target_filename) {
                    continue;
                }
                
                printf("\nProcessing: %s\n", dp->d_name);

                // Build the full path
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                // Read the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("  Could not read image %s\n", buffer);
                    continue;
                }
                
                // 1. DNN Feature Similarity
                float dnn_similarity = 0.0f;
                auto current_dnn_it = dnn_feature_database.find(current_filename);
                if(current_dnn_it == dnn_feature_database.end()) {
                    printf("  Warning: %s not found in CSV, using 0 for DNN score\n", current_filename.c_str());
                    dnn_similarity = 0.0f;
                } else {
                    // Using cosine similarity (convert distance to similarity: 1 - distance)
                    float dnn_distance = compute_cosine_distance(target_dnn_features, current_dnn_it->second);
                    dnn_similarity = 1.0f - dnn_distance;  // Convert distance to similarity
                }
                
                // 2. Color Histogram Intersection
                std::vector<std::vector<std::vector<float>>> color_hist = compute_3D_histogram(image, 8);
                float color_similarity = histogram_intersection_3D(target_color_hist, color_hist);
                
                // 3. Texture Histogram Intersection
                std::vector<std::vector<float>> texture_hist = compute_texture_histogram(image, 16, 16);
                float texture_similarity = histogram_intersection_2D(target_texture_hist, texture_hist);
                
                // Weighted combination (equal weights: 1/3 each)
                float w_dnn = 1.0f / 3.0f;
                float w_color = 1.0f / 3.0f;
                float w_texture = 1.0f / 3.0f;
                
                float combined_score = w_dnn * dnn_similarity + 
                                      w_color * color_similarity + 
                                      w_texture * texture_similarity;
                
                printf("  DNN: %.4f | Color: %.4f | Texture: %.4f | Combined: %.4f\n", 
                       dnn_similarity, color_similarity, texture_similarity, combined_score);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = combined_score;
                matches.push_back(match);
            }
        }

        // Sort matches (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\n=== Top %d LEAST Similar Matches ===\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\n=== Top %d MOST Similar Matches ===\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s\n", i + 1, matches[i].filename.c_str());
            printf("  Combined score: %.4f\n", matches[i].score);
        }
        
    } else if(method == "face-dnn-color-texture") {
        // Check if CSV file path is provided
        if(argc < 6 || std::string(argv[5]) == "--least-similar") {
            printf("Error: CSV file path required for face-dnn-color-texture method\n");
            printf("usage: %s <directory path> <target file path> [Top N matches] face-dnn-color-texture <csv file path> [--least-similar]\n", argv[0]);
            closedir(dirp);
            exit(-1);
        }
        
        std::string csv_path = argv[5];
        printf("=== Face-Filtered Hybrid DNN + Color + Texture Matching ===\n");
        printf("Only images with detected faces will be compared.\n");
        printf("Components:\n");
        printf("  1. DNN features (from CSV) - 33.3%% weight\n");
        printf("  2. Color histogram (RGB 8x8x8) - 33.3%% weight\n");
        printf("  3. Texture histogram (Gradient mag/angle 16x16) - 33.3%% weight\n");
        printf("Reading DNN features from: %s\n", csv_path.c_str());
        
        // Read all DNN features from CSV
        std::map<std::string, std::vector<float>> dnn_feature_database = read_features_from_csv(csv_path);
        
        if(dnn_feature_database.empty()) {
            printf("Error: Could not read features from CSV file or CSV is empty\n");
            closedir(dirp);
            exit(-1);
        }
        
        printf("Loaded %zu DNN feature vectors from CSV\n", dnn_feature_database.size());
        
        // ========================================
        // STEP 1: Check if target image has faces
        // ========================================
        cv::Mat target_grey;
        cv::cvtColor(target_image, target_grey, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> target_faces;
        detectFaces(target_grey, target_faces);
        
        printf("\n>>> Target image face detection: %zu face(s) found\n", target_faces.size());
        
        if(target_faces.empty()) {
            printf("WARNING: No faces detected in target image!\n");
            printf("This method is designed for face-based matching.\n");
            printf("Continuing anyway, but results may not be meaningful.\n\n");
        }
        
        // Get the target filename (basename without path)
        std::string target_path = argv[2];
        std::string target_filename = get_basename(target_path);
        
        // Find target DNN features in database
        auto target_dnn_it = dnn_feature_database.find(target_filename);
        if(target_dnn_it == dnn_feature_database.end()) {
            printf("Error: Target image '%s' not found in CSV feature database\n", target_filename.c_str());
            closedir(dirp);
            exit(-1);
        }
        
        std::vector<float> target_dnn_features = target_dnn_it->second;
        printf("Target image: %s (DNN feature dimension: %zu)\n", target_filename.c_str(), target_dnn_features.size());
        
        // Compute target color and texture features
        printf("Computing target color histogram...\n");
        std::vector<std::vector<std::vector<float>>> target_color_hist = compute_3D_histogram(target_image, 8);
        
        printf("Computing target texture histogram...\n");
        std::vector<std::vector<float>> target_texture_hist = compute_texture_histogram(target_image, 16, 16);
        
        // Vector to store matches
        std::vector<ImageMatch> matches;
        
        // Counters for statistics
        int images_processed = 0;
        int images_with_faces = 0;
        int images_without_faces = 0;
        
        // Loop over all the files in the image directory
        while((dp = readdir(dirp)) != NULL) {
            // Check if the file is an image
            if(strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

                std::string current_filename = dp->d_name;
                
                // Skip the target image itself
                if(current_filename == target_filename) {
                    continue;
                }
                
                images_processed++;
                printf("\nProcessing: %s\n", dp->d_name);

                // Build the full path
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                // Read the image
                cv::Mat image = cv::imread(buffer, cv::IMREAD_COLOR);
                if(image.empty()) {
                    printf("  Could not read image %s\n", buffer);
                    continue;
                }
                
                // ========================================
                // STEP 2: Detect faces - SKIP if no faces
                // ========================================
                cv::Mat greyFrame;
                cv::cvtColor(image, greyFrame, cv::COLOR_BGR2GRAY);
                std::vector<cv::Rect> faces;
                detectFaces(greyFrame, faces);
                
                printf("  Face detection: %zu face(s)\n", faces.size());
                
                // FILTER: Skip images without faces
                if(faces.empty()) {
                    printf("  >>> SKIPPING: No faces detected\n");
                    images_without_faces++;
                    continue;  // Skip to next image
                }
                
                images_with_faces++;
                printf("  >>> PROCESSING: Face(s) detected\n");
                
                // ========================================
                // STEP 3: Only compute features for images with faces
                // ========================================
                
                // 1. DNN Feature Similarity
                float dnn_similarity = 0.0f;
                auto current_dnn_it = dnn_feature_database.find(current_filename);
                if(current_dnn_it == dnn_feature_database.end()) {
                    printf("  Warning: %s not found in CSV, using 0 for DNN score\n", current_filename.c_str());
                    dnn_similarity = 0.0f;
                } else {
                    float dnn_distance = compute_cosine_distance(target_dnn_features, current_dnn_it->second);
                    dnn_similarity = 1.0f - dnn_distance;
                }
                
                // 2. Color Histogram Intersection
                std::vector<std::vector<std::vector<float>>> color_hist = compute_3D_histogram(image, 8);
                float color_similarity = histogram_intersection_3D(target_color_hist, color_hist);
                
                // 3. Texture Histogram Intersection
                std::vector<std::vector<float>> texture_hist = compute_texture_histogram(image, 16, 16);
                float texture_similarity = histogram_intersection_2D(target_texture_hist, texture_hist);
                
                // Weighted combination (equal weights: 1/3 each)
                float w_dnn = 1.0f / 3.0f;
                float w_color = 1.0f / 3.0f;
                float w_texture = 1.0f / 3.0f;
                
                float combined_score = w_dnn * dnn_similarity + 
                                    w_color * color_similarity + 
                                    w_texture * texture_similarity;
                
                printf("  DNN: %.4f | Color: %.4f | Texture: %.4f | Combined: %.4f\n", 
                    dnn_similarity, color_similarity, texture_similarity, combined_score);

                // Store the result
                ImageMatch match;
                match.filename = buffer;
                match.score = combined_score;
                matches.push_back(match);
            }
        }

        // Print statistics
        printf("\n=== Processing Statistics ===\n");
        printf("Total images scanned: %d\n", images_processed);
        printf("Images WITH faces (included): %d\n", images_with_faces);
        printf("Images WITHOUT faces (skipped): %d\n", images_without_faces);
        
        if(matches.empty()) {
            printf("\nNo images with faces found in the directory!\n");
            closedir(dirp);
            return 0;
        }

        // Sort matches
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\n=== Top %d LEAST Similar Matches (with faces) ===\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\n=== Top %d MOST Similar Matches (with faces) ===\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s\n", i + 1, matches[i].filename.c_str());
            printf("  Combined score: %.4f\n", matches[i].score);
        }
        
    } else if(method == "csv-cosine") {
        // Check if CSV file path is provided
        if(argc < 6 || std::string(argv[5]) == "--least-similar") {
            printf("Error: CSV file path required for csv-cosine method\n");
            printf("usage: %s <directory path> <target file path> [Top N matches] csv-cosine <csv file path> [--least-similar]\n", argv[0]);
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
        std::string target_filename = get_basename(target_path);
        
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
        
        // Sort by distance (ascending for most similar, descending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d LEAST similar matches (highest distance):\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d MOST similar matches (lowest distance):\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with cosine distance = %.4f\n", 
                   i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else if(method == "csv-cosine-similarity") {
        // Check if CSV file path is provided
        if(argc < 6 || std::string(argv[5]) == "--least-similar") {
            printf("Error: CSV file path required for csv-cosine-similarity method\n");
            printf("usage: %s <directory path> <target file path> [Top N matches] csv-cosine-similarity <csv file path> [--least-similar]\n", argv[0]);
            closedir(dirp);
            exit(-1);
        }
        
        std::string csv_path = argv[5];
        printf("Using CSV-based cosine similarity matching method\n");
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
        std::string target_filename = get_basename(target_path);
        
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
            
            // Compute cosine similarity
            float similarity = compute_cosine_similarity(target_features, features);
            
            printf("Comparing with %s: cosine similarity = %.4f\n", filename.c_str(), similarity);
            
            // Store the result
            ImageMatch match;
            match.filename = filename;
            match.score = similarity;
            matches.push_back(match);
        }
        
        // Sort by similarity (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches (lowest similarity):\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches (highest similarity):\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with cosine similarity = %.4f\n", 
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

        // Sort matches by SSD (ascending for most similar, descending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d LEAST similar matches (highest SSD):\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d MOST similar matches (lowest SSD):\n", top_n);
        }
        
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

        // Sort matches by intersection (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches (lowest intersection):\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches (highest intersection):\n", top_n);
        }
        
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

        // Sort matches by intersection (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
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

        // Sort matches by combined score (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
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

        // Sort matches by similarity (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
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

        // Sort matches by combined score (descending for most similar, ascending for least similar)
        if(show_least_similar) {
            std::sort(matches.begin(), matches.end(), compare_ascending);
            printf("\nTop %d LEAST similar matches:\n", top_n);
        } else {
            std::sort(matches.begin(), matches.end(), compare_descending);
            printf("\nTop %d MOST similar matches:\n", top_n);
        }
        
        for(int i = 0; i < top_n && i < (int)matches.size(); i++) {
            printf("Match %d: %s with combined score = %.4f\n", i + 1, matches[i].filename.c_str(), matches[i].score);
        }
        
    } else {
        printf("Unknown matching method: %s\n", method.c_str());
        printf("Available methods:\n");
        printf("  - ssd\n");
        printf("  - histogram\n");
        printf("  - multi-histogram\n");
        printf("  - color-texture\n");
        printf("  - gabor\n");
        printf("  - color-gabor\n");
        printf("  - csv-cosine\n");
        printf("  - csv-cosine-similarity\n");
        printf("  - hybrid-dnn-color-texture\n");
        printf("  - dominant-color-texture (NEW)\n");
        printf("  - edge-density (NEW)\n");
        printf("  - chi-square (NEW)\n");
        printf("  - hamming (NEW)\n");
        closedir(dirp);
        exit(-1);
    }

    printf("\nTerminating\n");
    closedir(dirp);  
    return 0;
}


// =======================================================================
// Functions for feature extraction
// =======================================================================
// Extract basename from path
std::string get_basename(const std::string &path) {
    size_t last_slash = path.find_last_of("/\\");
    return (last_slash != std::string::npos) ? path.substr(last_slash + 1) : path;
}

// Read feature vectors from CSV file
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
        // Check if first line is a header
        if(line.find("filename") != std::string::npos || line.find("image") != std::string::npos) {
            printf("Skipping header line\n");
        } else {
            // First line is data, process it
            file.seekg(0);
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

// Compute 3D histogram (RGB color space with 8 bins per channel)
std::vector<std::vector<std::vector<float>>> compute_3D_histogram(const cv::Mat &image, int bins) {
    std::vector<std::vector<std::vector<float>>> histogram(
        bins, 
        std::vector<std::vector<float>>(
            bins, 
            std::vector<float>(bins, 0.0f)
        )
    );
    
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            
            int r_bin = (pixel[2] * bins) / 256;
            int g_bin = (pixel[1] * bins) / 256;
            int b_bin = (pixel[0] * bins) / 256;
            
            r_bin = std::min(r_bin, bins - 1);
            g_bin = std::min(g_bin, bins - 1);
            b_bin = std::min(b_bin, bins - 1);
            
            histogram[r_bin][g_bin][b_bin]++;
        }
    }

    float sum = 0.0f;
    for(const auto &plane : histogram) {
        for(const auto &row : plane) {
            for(float value : row) {
                sum += value;
            }
        }
    }

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

// Compute 3D histogram for a specific row region
std::vector<std::vector<std::vector<float>>> compute_3D_histogram_region(
    const cv::Mat &image, int bins, int start_row, int end_row) {
    
    std::vector<std::vector<std::vector<float>>> histogram(
        bins, 
        std::vector<std::vector<float>>(
            bins, 
            std::vector<float>(bins, 0.0f)
        )
    );
    
    for(int i = start_row; i < end_row; i++) {
        for(int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            
            int r_bin = (pixel[2] * bins) / 256;
            int g_bin = (pixel[1] * bins) / 256;
            int b_bin = (pixel[0] * bins) / 256;
            
            r_bin = std::min(r_bin, bins - 1);
            g_bin = std::min(g_bin, bins - 1);
            b_bin = std::min(b_bin, bins - 1);
            
            histogram[r_bin][g_bin][b_bin]++;
        }
    }

    float sum = 0.0f;
    for(const auto &plane : histogram) {
        for(const auto &row : plane) {
            for(float value : row) {
                sum += value;
            }
        }
    }

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
    std::vector<std::vector<float>> histogram(mag_bins, std::vector<float>(angle_bins, 0.0f));
    
    cv::Mat gray;
    if(image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    
    for(int i = 0; i < gray.rows; i++) {
        for(int j = 0; j < gray.cols; j++) {
            float gx = grad_x.at<float>(i, j);
            float gy = grad_y.at<float>(i, j);
            
            float magnitude = std::sqrt(gx * gx + gy * gy);
            float angle = std::atan2(gy, gx) * 180.0f / M_PI;
            if(angle < 0) angle += 360.0f;
            
            int mag_bin = static_cast<int>((magnitude * mag_bins) / 256.0f);
            mag_bin = std::min(mag_bin, mag_bins - 1);
            
            int angle_bin = static_cast<int>((angle * angle_bins) / 360.0f);
            angle_bin = std::min(angle_bin, angle_bins - 1);
            
            histogram[mag_bin][angle_bin]++;
        }
    }
    
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
            double x_theta = x * std::cos(theta) + y * std::sin(theta);
            double y_theta = -x * std::sin(theta) + y * std::cos(theta);
            
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
    
    cv::Mat gray;
    if(image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F);
    
    int ksize = 21;
    double sigma = 5.0;
    double gamma = 0.5;
    double psi = 0;
    
    std::vector<double> lambdas = {5.0, 10.0, 15.0, 20.0};
    std::vector<double> thetas = {0, M_PI/4, M_PI/2, 3*M_PI/4};
    
    for(double lambda : lambdas) {
        for(double theta : thetas) {
            cv::Mat kernel = create_gabor_kernel(ksize, sigma, theta, lambda, gamma, psi);
            cv::Mat filtered;
            cv::filter2D(gray_float, filtered, CV_32F, kernel);
            
            cv::Scalar mean, stddev;
            cv::meanStdDev(filtered, mean, stddev);
            
            features.push_back(mean[0]);
            features.push_back(stddev[0]);
        }
    }
    
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

// Extract dominant colors using K-means clustering
std::vector<cv::Vec3f> extract_dominant_colors_kmeans(const cv::Mat &image, int k) {
    std::vector<cv::Vec3f> dominant_colors;
    
    // Reshape image to a list of pixels
    cv::Mat reshaped = image.reshape(1, image.rows * image.cols);
    cv::Mat samples;
    reshaped.convertTo(samples, CV_32F);
    
    // K-means clustering
    cv::Mat labels;
    cv::Mat centers;
    int attempts = 3;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.2);
    
    cv::kmeans(samples, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);
    
    // Extract cluster centers as dominant colors
    for(int i = 0; i < k; i++) {
        cv::Vec3f color(centers.at<float>(i, 0), 
                       centers.at<float>(i, 1), 
                       centers.at<float>(i, 2));
        dominant_colors.push_back(color);
    }
    
    return dominant_colors;
}

// Compute edge density map by dividing image into grid
std::vector<float> compute_edge_density_map(const cv::Mat &image, int grid_size) {
    std::vector<float> edge_density;
    
    // Convert to grayscale
    cv::Mat gray;
    if(image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Compute edges using Canny
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    // Divide into grid and compute density for each cell
    int cell_height = gray.rows / grid_size;
    int cell_width = gray.cols / grid_size;
    
    for(int i = 0; i < grid_size; i++) {
        for(int j = 0; j < grid_size; j++) {
            int start_y = i * cell_height;
            int start_x = j * cell_width;
            int end_y = (i == grid_size - 1) ? gray.rows : (i + 1) * cell_height;
            int end_x = (j == grid_size - 1) ? gray.cols : (j + 1) * cell_width;
            
            // Extract cell
            cv::Rect cell_rect(start_x, start_y, end_x - start_x, end_y - start_y);
            cv::Mat cell = edges(cell_rect);
            
            // Count edge pixels
            int edge_count = cv::countNonZero(cell);
            int total_pixels = cell.rows * cell.cols;
            
            // Compute density (percentage of edge pixels)
            float density = static_cast<float>(edge_count) / total_pixels;
            edge_density.push_back(density);
        }
    }
    
    return edge_density;
}

// =======================================================================
// Similarity and distance metrics
// ======================================================================

// Compute Sum of Squared Differences
float compute_ssd(const std::vector<float> &a, const std::vector<float> &b) {
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

// Compute cosine similarity
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

// Compute cosine distance
float compute_cosine_distance(const std::vector<float> &a, const std::vector<float> &b) {
    float similarity = compute_cosine_similarity(a, b);
    return 1.0f - similarity;
}

// Compute histogram intersection for 2D histograms
float histogram_intersection_2D(const std::vector<std::vector<float>> &hist1,
                                const std::vector<std::vector<float>> &hist2) {
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

// Compute histogram intersection for 3D histograms
float histogram_intersection_3D(const std::vector<std::vector<std::vector<float>>> &hist1, 
                                const std::vector<std::vector<std::vector<float>>> &hist2) {
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

// Compute chi-square distance for 3D histograms
float compute_chi_square_distance_3D(const std::vector<std::vector<std::vector<float>>> &hist1,
                                     const std::vector<std::vector<std::vector<float>>> &hist2) {
    if(hist1.size() != hist2.size() || 
       hist1[0].size() != hist2[0].size() || 
       hist1[0][0].size() != hist2[0][0].size()) {
        printf("Error: Histogram dimensions don't match\n");
        return std::numeric_limits<float>::max();
    }
    
    float chi_square = 0.0f;
    const float epsilon = 1e-10f; // Avoid division by zero
    
    for(size_t i = 0; i < hist1.size(); i++) {
        for(size_t j = 0; j < hist1[i].size(); j++) {
            for(size_t k = 0; k < hist1[i][j].size(); k++) {
                float sum = hist1[i][j][k] + hist2[i][j][k];
                if(sum > epsilon) {
                    float diff = hist1[i][j][k] - hist2[i][j][k];
                    chi_square += (diff * diff) / sum;
                }
            }
        }
    }
    
    return chi_square;
}

// Compute chi-square distance for 2D histograms
float compute_chi_square_distance_2D(const std::vector<std::vector<float>> &hist1,
                                     const std::vector<std::vector<float>> &hist2) {
    if(hist1.size() != hist2.size() || hist1[0].size() != hist2[0].size()) {
        printf("Error: 2D Histogram dimensions don't match\n");
        return std::numeric_limits<float>::max();
    }
    
    float chi_square = 0.0f;
    const float epsilon = 1e-10f;
    
    for(size_t i = 0; i < hist1.size(); i++) {
        for(size_t j = 0; j < hist1[i].size(); j++) {
            float sum = hist1[i][j] + hist2[i][j];
            if(sum > epsilon) {
                float diff = hist1[i][j] - hist2[i][j];
                chi_square += (diff * diff) / sum;
            }
        }
    }
    
    return chi_square;
}

// Compute chi-square distance for 1D vectors
float compute_chi_square_distance_1D(const std::vector<float> &hist1, const std::vector<float> &hist2) {
    if(hist1.size() != hist2.size()) {
        printf("Error: Vector dimensions don't match\n");
        return std::numeric_limits<float>::max();
    }
    
    float chi_square = 0.0f;
    const float epsilon = 1e-10f;
    
    for(size_t i = 0; i < hist1.size(); i++) {
        float sum = hist1[i] + hist2[i];
        if(sum > epsilon) {
            float diff = hist1[i] - hist2[i];
            chi_square += (diff * diff) / sum;
        }
    }
    
    return chi_square;
}

// Compute Hamming distance between two feature vectors (after binarization)
float compute_hamming_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if(a.size() != b.size()) {
        printf("Error: Feature vectors have different sizes\n");
        return std::numeric_limits<float>::max();
    }
    
    // Compute mean for binarization threshold
    float mean_a = 0.0f, mean_b = 0.0f;
    for(size_t i = 0; i < a.size(); i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= a.size();
    mean_b /= b.size();
    
    // Count differing bits
    float hamming = 0.0f;
    for(size_t i = 0; i < a.size(); i++) {
        bool bit_a = a[i] > mean_a;
        bool bit_b = b[i] > mean_b;
        if(bit_a != bit_b) {
            hamming++;
        }
    }
    
    return hamming;
}


/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat embedding  holds the embedding vector after the function returns
  cv::dnn::Net net   the pre-trained ResNet 18 network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug ) {
  const int ORNet_size = 224;
  cv::Mat blob;

  // have the function do the ImageNet mean and SD normalization
  // the function also scales the image to 224 x 224
  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) * (1/0.226), // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  cv::Scalar( 124, 116, 104),  // subtract mean prior to scaling
			  true,   // swapRB
			  false,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!resnetv22_flatten0_reshape0" ); // the name of the embedding layer to grab

  if(debug) {
    cv::imshow( "src", src );
    std::cout << embedding << std::endl;
    std::cout << embedding.rows << " " << embedding.cols << std::endl;
    cv::waitKey(0);
  }

  return(0);
}
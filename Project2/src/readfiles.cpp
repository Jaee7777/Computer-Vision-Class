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
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include "../include/csv_util.h"

std::vector<float> extract_feature_vector(const cv::Mat &image);
float compute_ssd(const std::vector<float> &a, const std::vector<float> &b);
struct ImageMatch {
    std::string filename;
    float ssd;
};
bool compare_by_ssd(const ImageMatch &a, const ImageMatch &b) {
    return a.ssd < b.ssd;
}

/*
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if( argc < 4) {
    printf("usage: %s <directory path> <target file path> [Top N matches]\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // find feature vector for the target image
  cv::Mat target_image = cv::imread( argv[2], cv::IMREAD_COLOR );
  if( target_image.empty() ) {
    printf("Could not read target image %s\n", argv[2]);
    exit(-1);
  }
  std::vector<float> target_feature_vector = extract_feature_vector(target_image);

  // vector to store file names and SSD
  std::vector<ImageMatch> matches;

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);

      // find feature vector for the image
      cv::Mat image = cv::imread( buffer, cv::IMREAD_COLOR );
      if( image.empty() ) {
        printf("Could not read image %s\n", buffer);
        exit(-1);
      }
      std::vector<float> feature_vector = extract_feature_vector(image);

      // find SSD
      float ssd = compute_ssd(target_feature_vector, feature_vector);
      printf("SSD of image with target: %.2f\n", ssd);

      // store the result
      ImageMatch match;
      match.filename = buffer;
      match.ssd = ssd;
      matches.push_back(match);
    }
  }

  closedir(dirp);  

  // sort matches by SSD
  std::sort(matches.begin(), matches.end(), compare_by_ssd);
  int top_n = std::stoi(argv[3]);
  for (int i = 0; i < top_n && i < (int)matches.size(); i++) {
      printf("Match %d: %s with SSD = %.2f\n", i + 1, matches[i].filename.c_str(), matches[i].ssd);
  }

  printf("Terminating\n");

  return(0);
}

// Use float instead of uchar since we need to compute SSD
std::vector<float> extract_feature_vector(const cv::Mat &image) {
    std::vector<float> feature_vector;
    
    // Check size of image
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
  // safety check
  if (a.size() != b.size()) {
    printf("Error: Feature vectors have different sizes\n");
    return -1.0f;
  }
    
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}
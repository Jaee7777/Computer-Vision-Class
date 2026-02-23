/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 3.

  Skeleton code provided on the assignment page (Project 1) was used as a template.

  OpenCV Documentation was the final source for verification of each function.
  (https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
  AI Overview of Google was used to find related functions.
  Claude AI was used for code review and debugging.

*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "../include/filter.h"
#include "../include/faceDetect.h"
#include "../include/DA2Network.hpp"

// Magic numbers for the analysis
const int MIN_AREA     = 500;
const int AXIS_LENGTH  = 60;
const int MORPH_RADIUS = 1;
const int KMEANS_K     = 2;
const int KMEANS_EVERY = 30;

int main(int argc, char *argv[]) {
    cv::VideoCapture capdev(0);
    if( !capdev.isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev.get(cv::CAP_PROP_FRAME_WIDTH ),
		(int) capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // initialization
    cv::Mat frame, sat, blurred, thresh, binary_erode, binary_dilate,
		binary_open, binary_close,frame_region, frame_feature;
    int threshVal = 80;
    int frameCount = 0;

    cv::Mat labels, stats, centroids;

    // Initialize DA2Network for depth estimation
    // DA2Network da_net("model/model_fp16.onnx");

	for(;;) {
		capdev >> frame; // get a new frame from the camera, treat as a stream
		if (frame.empty()) {
			printf("frame is empty\n");
			break;
		}   

		frame.copyTo(frame_region);
		frame.copyTo(frame_feature);

		saturation(frame, sat);
		blur5x5_grey(sat, blurred);
		// Apply k=2 clustering on every 30 frames
		if (frameCount % KMEANS_EVERY == 0) {
			threshVal = kmeansThreshold(blurred, KMEANS_K);
		}
		threshold(blurred, thresh, threshVal);
		// Using radius=1 for 3x3 kernels of morphological filters
		// erode(thresh, binary_erode, MORPH_RADIUS);
		// dilate(thresh, binary_dilate, MORPH_RADIUS);
		// opening(thresh, binary_open, MORPH_RADIUS);
		closing(thresh, binary_close, MORPH_RADIUS);

		int numComponents = cv::connectedComponentsWithStats(binary_close, labels, stats, centroids, 8);

		cv::imshow("Original", frame);
		cv::imshow("Saturation", sat);
		cv::imshow("Thresholded", thresh);
		// cv::imshow("Eroded", binary_erode);
		// cv::imshow("Dilated", binary_dilate);
		// cv::imshow("Opening", binary_open);
		cv::imshow("Closing", binary_close);
		
		for (int i = 1; i < numComponents; ++i) {
			if (stats.at<int>(i, cv::CC_STAT_AREA) < MIN_AREA) continue; // Remove small regions

			// ============== Centroid and bounding box of the region ===========
			// Geometry of regions.
			int x = stats.at<int>(i, cv::CC_STAT_LEFT);
			int y = stats.at<int>(i, cv::CC_STAT_TOP);
			int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
			int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			double cx = centroids.at<double>(i, 0);
			double cy = centroids.at<double>(i, 1);
			
			cv::Point center((int)cx, (int)cy); // Centroid of the region
			cv::rectangle(frame_region, cv::Point(x, y), cv::Point(x+w, y+h),
						cv::Scalar(0, 255, 0), 2);  // green box for region
			cv::circle(frame_region, center, 5,
					cv::Scalar(0, 0, 255), -1);     // red dot for centroid
			cv::putText(frame_region, std::to_string(i),   // component ID label
						cv::Point(x, y - 5),
						cv::FONT_HERSHEY_SIMPLEX, 0.5,
						cv::Scalar(255, 255, 0), 1);

			// =========Axis of the least central moment and bounding box of the region =========
			cv::Mat mask = (labels == i);
			cv::Moments m = cv::moments(mask, true);
			std::vector<cv::Point> points;
			cv::findNonZero(mask, points);
			cv::RotatedRect rrect = cv::minAreaRect(points);
			cv::Point2f corners[4];
			rrect.points(corners);

			// Axis of the least central moment
			double angle = rrect.angle * CV_PI / 180.0;
			cv::Point offset(AXIS_LENGTH*cos(angle), AXIS_LENGTH*sin(angle));
			cv::line(frame_feature,
					cv::Point(center.x - offset.x, center.y - offset.y),
					cv::Point(center.x + offset.x, center.y + offset.y),
					cv::Scalar(255, 0, 0), 2);

			// Bounding box
			for (int k = 0; k < 4; ++k)
				cv::line(frame_feature, corners[k], corners[(k+1) % 4],
						cv::Scalar(0, 255, 255), 2);
		}

		cv::imshow("Region Map", frame_region);
		cv::imshow("Feature of regions", frame_feature);

		// see if there is a waiting keystroke
		int key = cv::waitKey(10);

		if( key == 'q') {
			break;
		}
		else if (key == 's') {
			std::string filename_1 = "frame_thresh_" + std::to_string(frameCount) + ".jpg";
			cv::imwrite(filename_1, thresh);
			std::string filename_2 = "frame_clean_" + std::to_string(frameCount) + ".jpg";
			cv::imwrite(filename_2, binary_close);
			frameCount++;
		}
	}

	return(0);
}

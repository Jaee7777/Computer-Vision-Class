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
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <algorithm>

#include "../include/filter.h"
#include "../include/utilities.h"
// #include "features.h"
// #include "vision.h"

// Magic numbers for the analysis
const int MIN_AREA     = 1500;
const int AXIS_LENGTH  = 100;
const int MORPH_RADIUS = 1;
const int KMEANS_K     = 2;
const int KMEANS_EVERY = 30;


int main(int argc, char *argv[]) {
    cv::VideoCapture capdev(0);
    if( !capdev.isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    cv::Size refS( (int) capdev.get(cv::CAP_PROP_FRAME_WIDTH ),
		(int) capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::Mat frame, sat, blurred, thresh, binary_erode, binary_dilate,
		binary_open, binary_close, frame_region, frame_feature;
    int threshVal = 80;
    int frameCount = 0;
    int saveCount  = 0;

    cv::Mat labels, stats, centroids;

    // ===== 'h' = Hu moments (default), 'n' = neural embedding =====
    bool useEmbedding = false;

    // ===== Hu moment training data =====
	std::vector<std::vector<float>> trainingData;
	std::vector<std::string> trainingLabels;
	std::ifstream file("data/training_data.csv");
	if (!file.is_open()) {
		printf("Warning: Could not open training_data.csv for reading\n");
	} else {
		std::string line;
		std::getline(file, line); // skip header
		while (std::getline(file, line)) {
			std::stringstream ss(line);
			std::string token;
			std::vector<float> row;
			std::getline(ss, token, ',');
			trainingLabels.push_back(token);
			while (std::getline(ss, token, ','))
				row.push_back(std::stof(token));
			trainingData.push_back(row);
		}
	}

    // ===== Embedding training data =====
    std::vector<std::vector<float>> embTrainingData;
    std::vector<std::string> embTrainingLabels;
    std::ifstream embFile("data/embedding_data.csv");
    if (!embFile.is_open()) {
        printf("Warning: Could not open embedding_data.csv for reading\n");
    } else {
        std::string line;
        std::getline(embFile, line); // skip header
        while (std::getline(embFile, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<float> row;
            std::getline(ss, token, ',');
            embTrainingLabels.push_back(token);
            while (std::getline(ss, token, ','))
                row.push_back(std::stof(token));
            embTrainingData.push_back(row);
        }
    }

	std::map<int, std::string> componentLabels;
	bool classifyMode = false; // Classification mode on/off

	// ===== Confusion matrix data =====
	std::map<std::string, std::map<std::string, int>> confusionMatrix;
	std::vector<std::string> allLabels;

	// ===== Load ResNet18 =====
	cv::dnn::Net onnxNet = cv::dnn::readNet("model/resnet18-v2-7.onnx");
	if (onnxNet.empty()) {
		printf("Error: could not load ResNet model\n");
		return -1;
	}
    printf("ResNet18 loaded successfully\n");
    printf("Controls: 't'=train  'c'=toggle classify  'e'=evaluate  'n'=toggle Hu/Embedding  's'=save  'q'=quit\n");

	for(;;) {
		capdev >> frame;
		if (frame.empty()) {
			printf("frame is empty\n");
			break;
		}

		frameCount++;
		int key = cv::waitKey(10);

		// Toggle classification mode
		if (key == 'c') {
			classifyMode = !classifyMode;
			if (!classifyMode) componentLabels.clear();
			printf("Classification mode: %s\n", classifyMode ? "ON" : "OFF");
		}

        // Toggle between Hu moments and embedding
        if (key == 'n') {
            useEmbedding = !useEmbedding;
            componentLabels.clear();
            printf("Method: %s\n", useEmbedding ? "Neural Embedding" : "Hu Moments");
        }

		frame.copyTo(frame_region);
		frame.copyTo(frame_feature);

		saturation(frame, sat);
		blur5x5_grey(sat, blurred);
		if (frameCount % KMEANS_EVERY == 0) {
			threshVal = kmeansThreshold(blurred, KMEANS_K);
		}
		threshold(blurred, thresh, threshVal);
		closing(thresh, binary_close, MORPH_RADIUS);

		int numComponents = cv::connectedComponentsWithStats(binary_close, labels, stats, centroids, 8);

		cv::imshow("Original", frame);
		cv::imshow("Saturation", sat);
		cv::imshow("Thresholded", thresh);
		cv::imshow("Closing", binary_close);

		for (int i = 1; i < numComponents; ++i) {
			if (stats.at<int>(i, cv::CC_STAT_AREA) < MIN_AREA) continue;

			// ===== Centroid and bounding box =====
			int x = stats.at<int>(i, cv::CC_STAT_LEFT);
			int y = stats.at<int>(i, cv::CC_STAT_TOP);
			int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
			int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			double cx = centroids.at<double>(i, 0);
			double cy = centroids.at<double>(i, 1);

			cv::Point center((int)cx, (int)cy);
			cv::rectangle(frame_region, cv::Point(x, y), cv::Point(x+w, y+h),
						cv::Scalar(0, 255, 0), 2);
			cv::circle(frame_region, center, 5, cv::Scalar(0, 0, 255), -1);
			cv::putText(frame_region, std::to_string(i),
						cv::Point(x, y - 5),
						cv::FONT_HERSHEY_SIMPLEX, 0.8,
						cv::Scalar(255, 255, 0), 1);

			// ===== Axis of least central moment and oriented bounding box =====
			cv::Mat mask = (labels == i);
			cv::Moments m = cv::moments(mask, true);
			std::vector<cv::Point> points;
			cv::findNonZero(mask, points);
			cv::RotatedRect rrect = cv::minAreaRect(points);
			cv::Point2f corners[4];
			rrect.points(corners);
			double angle = rrect.angle * CV_PI / 180.0;

			cv::Point offset(AXIS_LENGTH*cos(angle), AXIS_LENGTH*sin(angle));
			cv::line(frame_feature,
					cv::Point(center.x - offset.x, center.y - offset.y),
					cv::Point(center.x + offset.x, center.y + offset.y),
					cv::Scalar(255, 0, 0), 2);
			for (int k = 0; k < 4; ++k)
				cv::line(frame_feature, corners[k], corners[(k+1) % 4],
						cv::Scalar(0, 255, 255), 2);

			double area = stats.at<int>(i, cv::CC_STAT_AREA);

			// ===== Hu moment feature extractor =====
			auto extractHuFeatures = [&]() -> std::vector<float> {
				double hu[7];
				cv::HuMoments(m, hu);
				for (int j = 0; j < 7; j++)
					hu[j] = copysign(log10(abs(hu[j]) + 1e-10), hu[j]);
				double aspect = (double)w / h;
				double extent = area / (w * h);
				return {
					(float)hu[0], (float)hu[1], (float)hu[2], (float)hu[3],
					(float)hu[4], (float)hu[5], (float)hu[6],
					(float)aspect, (float)extent, (float)area
				};
			};

			// ===== Embedding extractor =====
			auto extractEmbedding = [&]() -> std::vector<float> {
				// Project region points onto primary/secondary axes
				float minE1=1e9, maxE1=-1e9, minE2=1e9, maxE2=-1e9;
				for (auto &pt : points) {
					float dx = pt.x - cx, dy = pt.y - cy;
					float e1 =  dx*cos(angle) + dy*sin(angle);
					float e2 = -dx*sin(angle) + dy*cos(angle);
					minE1 = std::min(minE1, e1); maxE1 = std::max(maxE1, e1);
					minE2 = std::min(minE2, e2); maxE2 = std::max(maxE2, e2);
				}
				cv::Mat embimage, embedding;
				prepEmbeddingImage(frame, embimage, (int)cx, (int)cy,
								   angle, minE1, maxE1, minE2, maxE2, 0);
				if (embimage.empty()) return {};
				getEmbedding(embimage, embedding, onnxNet, 0);
				return std::vector<float>(embedding.begin<float>(), embedding.end<float>());
			};

			// ===== Classification helper =====
			auto classify = [&](const std::vector<float> &query,
								const std::vector<std::vector<float>> &db,
								const std::vector<std::string> &dbLabels,
								bool usesCosine) -> std::string {
				if (db.empty()) return "unknown";
				float minDist = std::numeric_limits<float>::max();
				std::string best = "unknown";
				for (int z = 0; z < (int)db.size(); z++) {
					if (db[z].size() != query.size()) {
						printf("Skipping row %d: size mismatch (%zu vs %zu)\n",
							z, query.size(), db[z].size());
						continue;
					}
					float dist = usesCosine ? cosineDist(query, db[z])
											: compute_ssd(query, db[z]);
					if (dist >= 0.0f && dist < minDist) {
						minDist = dist;
						best = dbLabels[z];
					}
				}
				return best;
			};

			// ===== Training ('t') =====
			if (key == 't') {
				std::string manual_label;
				std::cout << "Enter label for this region: ";
				std::cin >> manual_label;

				if (useEmbedding) {
					std::vector<float> emb = extractEmbedding();
					if (emb.empty()) { printf("Embedding extraction failed\n"); continue; }

					std::ifstream check("data/embedding_data.csv");
					bool isEmpty = !check.good() ||
						check.peek() == std::ifstream::traits_type::eof();
					check.close();
					std::ofstream csv("data/embedding_data.csv", std::ios::app);
					if (isEmpty) {
						csv << "label";
						for (int j = 0; j < 512; j++) csv << ",e" << j;
						csv << "\n";
					}
					csv << manual_label;
					for (float f : emb) csv << "," << f;
					csv << "\n";
					csv.close();
					embTrainingData.push_back(emb);
					embTrainingLabels.push_back(manual_label);
					printf("Saved embedding for '%s'\n", manual_label.c_str());
				} else {
					std::vector<float> features = extractHuFeatures();

					std::ifstream check("data/training_data.csv");
					bool isEmpty = !check.good() ||
						check.peek() == std::ifstream::traits_type::eof();
					check.close();
					std::ofstream csv("data/training_data.csv", std::ios::app);
					if (isEmpty)
						csv << "label,hu0,hu1,hu2,hu3,hu4,hu5,hu6,aspect,extent,area\n";
					csv << manual_label;
					for (float f : features) csv << "," << f;
					csv << "\n";
					csv.close();
					trainingData.push_back(features);
					trainingLabels.push_back(manual_label);
					printf("Saved Hu features for '%s'\n", manual_label.c_str());
				}
			}

			// ===== Evaluate ('e') =====
			if (key == 'e') {
				std::vector<float> query = useEmbedding ? extractEmbedding()
														: extractHuFeatures();
				if (query.empty()) { printf("Feature extraction failed\n"); continue; }

				auto &db     = useEmbedding ? embTrainingData  : trainingData;
				auto &dbLbls = useEmbedding ? embTrainingLabels : trainingLabels;
				if (db.empty()) { printf("No training data loaded.\n"); continue; }

				std::string predicted = classify(query, db, dbLbls, useEmbedding);

				std::string trueLabel;
				std::cout << "Enter TRUE label (predicted: " << predicted << "): ";
				std::cin >> trueLabel;

				confusionMatrix[trueLabel][predicted]++;
				if (std::find(allLabels.begin(), allLabels.end(), trueLabel) == allLabels.end())
					allLabels.push_back(trueLabel);
				if (std::find(allLabels.begin(), allLabels.end(), predicted) == allLabels.end())
					allLabels.push_back(predicted);
				printf("Recorded: true='%s' predicted='%s'\n", trueLabel.c_str(), predicted.c_str());
			}

			// ===== Classification =====
			if (classifyMode) {
				std::vector<float> query = useEmbedding ? extractEmbedding()
														: extractHuFeatures();
				if (query.empty()) continue;

				auto &db     = useEmbedding ? embTrainingData  : trainingData;
				auto &dbLbls = useEmbedding ? embTrainingLabels : trainingLabels;
				if (db.empty()) continue;

				componentLabels[i] = classify(query, db, dbLbls, useEmbedding);
			}

			// ===== Draw classification label =====
			if (componentLabels.count(i)) {
				// Show method indicator alongside label
				std::string display = componentLabels[i] +
					(useEmbedding ? " [emb]" : " [hu]");
				cv::putText(frame_feature, display,
							cv::Point(x, y - 5),
							cv::FONT_HERSHEY_SIMPLEX, 0.8,
							cv::Scalar(0, 255, 0), 2);
			}
		}

		cv::imshow("Region Map", frame_region);
		cv::imshow("Feature of regions", frame_feature);

		if (key == 'q') {
			if (!confusionMatrix.empty()) {
				std::sort(allLabels.begin(), allLabels.end());
				printf("\n===== Confusion Matrix =====\n");
				printf("Rows = True label, Cols = Predicted label\n\n");
				printf("%15s", "");
				for (auto &col : allLabels) printf("%15s", col.c_str());
				printf("\n");
				for (auto &row : allLabels) {
					printf("%15s", row.c_str());
					for (auto &col : allLabels)
						printf("%15d", confusionMatrix[row][col]);
					printf("\n");
				}
				std::ofstream cm("data/confusion_matrix.csv");
				cm << "true\\predicted";
				for (auto &col : allLabels) cm << "," << col;
				cm << "\n";
				for (auto &row : allLabels) {
					cm << row;
					for (auto &col : allLabels)
						cm << "," << confusionMatrix[row][col];
					cm << "\n";
				}
				cm.close();
				printf("\nConfusion matrix saved to data/confusion_matrix.csv\n");
			}
			break;
		} else if (key == 's') {
			cv::imwrite("frame_thresh_"  + std::to_string(saveCount) + ".jpg", thresh);
			cv::imwrite("frame_clean_"   + std::to_string(saveCount) + ".jpg", binary_close);
			cv::imwrite("frame_feature_" + std::to_string(saveCount) + ".jpg", frame_feature);
			cv::imwrite("frame_region_"  + std::to_string(saveCount) + ".jpg", frame_region);
			saveCount++;
		}
	}

	return(0);
}
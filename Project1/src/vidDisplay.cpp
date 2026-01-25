/*
  Jaee Oh
  Spring 2026

  CS 5330 Computer Vision

  Project 1.

  Skeleton code provided on the assignment page was used as a template.

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

int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        int frameCount = 0;

        // Initialize DA2Network for depth estimation
        DA2Network da_net("model/model_fp16.onnx");

        // Initialize for invisibility cloak
        cv::Mat background;
        bool cloakMode = false;
        int cloakThreshold = 80;

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }                
                cv::imshow("Video", frame);

                // Invisibility cloak effect (runs continuously when enabled)
                if (cloakMode && !background.empty()) {
                    // Detect faces
                    cv::Mat greyFrame;
                    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                    std::vector<cv::Rect> faces;
                    detectFaces(greyFrame, faces);

                    // Start with background (everything invisible)
                    cv::Mat result = background.clone();

                    // Copy only faces from current frame
                    for (size_t i = 0; i < faces.size(); i++) {
                        frame(faces[i]).copyTo(result(faces[i]));
                    }

                    cv::imshow("Floating Face", result);
                }

                // see if there is a waiting keystroke
                char key = cv::waitKey(10);

                if( key == 'q') {
                    break;
                }
                else if (key == 's') {
                    std::string filename = "frame_" + std::to_string(frameCount) + ".jpg";
                    cv::imwrite(filename, frame);
                    frameCount++;
                }
                else if (key == 'g') {
                    cv::Mat greyFrame;
                    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                    cv::imshow("Greyscale Image", greyFrame);
                }
                else if (key == 'h') {
                    cv::Mat greyFrame_custom;
                    greyscale(frame, greyFrame_custom);
                    cv::imshow("Greyscale Image - Average", greyFrame_custom);
                }
                else if (key == 'j') {
                    cv::Mat sepiaFrame;
                    sepia(frame, sepiaFrame);
                    cv::imshow("Sepia Image", sepiaFrame);
                }
                else if (key == 'b') {
                    cv::Mat blurFrame;
                    blur5x5_2(frame, blurFrame);
                    cv::imshow("Blurred Image", blurFrame);
                }
                else if (key == 'x') {
                    cv::Mat sobelXFrame, displayX;
                    sobelX3x3(frame, sobelXFrame);
                    cv::convertScaleAbs(sobelXFrame, displayX); // Convert to 8-bit for display.
                    cv::imshow("Sobel X Image", displayX);
                }
                else if (key == 'y') {
                    cv::Mat sobelYFrame, displayY;
                    sobelY3x3(frame, sobelYFrame);
                    cv::convertScaleAbs(sobelYFrame, displayY); // Convert to 8-bit for display.
                    cv::imshow("Sobel Y Image", displayY);
                }
                else if (key == 'm') {
                    cv::Mat sobelXFrame, sobelYFrame, magnitudeFrame;
                    sobelX3x3(frame, sobelXFrame);
                    sobelY3x3(frame, sobelYFrame);
                    magnitude(sobelXFrame, sobelYFrame, magnitudeFrame);
                    cv::imshow("Sobel Magnitude Image", magnitudeFrame);
                }
                else if (key == 'i') {
                    cv::Mat blurQuantFrame;
                    int levels = 7; // Quantization level.
                    blurQuantize(frame, blurQuantFrame, levels);
                    cv::imshow("Blured and Quantized Image", blurQuantFrame);
                }
                else if (key == 'f') {
                    cv::Mat greyFrame;
                    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                    std::vector<cv::Rect> faces;
                    detectFaces(greyFrame, faces);
                    for (size_t i = 0; i < faces.size(); i++) {
                        cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
                    }
                    cv::imshow("Face Detection", frame);
                }
                else if (key == 'd') {
                    float scale_factor = 512.0 / (frame.rows > frame.cols ? frame.cols : frame.rows);
                    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

                    da_net.set_input( frame, scale_factor );

                    cv::Mat depthFrame;
                    da_net.run_network( depthFrame, frame.size() );

                    cv::Mat depthVis;
                    cv::applyColorMap(depthFrame, depthVis, cv::COLORMAP_INFERNO );

                    cv::imshow("Depth Image", depthVis);
                }
                else if (key == 'z') {
                    float scale_factor = 512.0 / (frame.rows > frame.cols ? frame.cols : frame.rows);
                    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

                    da_net.set_input( frame, scale_factor );

                    cv::Mat depthFrame;
                    da_net.run_network( depthFrame, frame.size() );

                    cv::Mat result = frame.clone();

                    for (int y = 0; y < frame.rows; y++) {
                        for (int x = 0; x < frame.cols; x++) {
                            uchar depth = depthFrame.at<uchar>(y, x);
                            cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                            
                            if (depth < 85) {
                                // warm tint (red/orange)
                                pixel[2] = cv::min(255, pixel[2] + 50);
                            } else if (depth < 170) {
                                // green tint
                                pixel[1] = cv::min(255, pixel[1] + 50);
                            } else {
                                // cool tint (blue)
                                pixel[0] = cv::min(255, pixel[0] + 50);
                            }
                        }
                    }

                    cv::imshow("Depth Image", result);
                }
                else if (key == 'e') {
                    cv::Mat result = frame ^ cv::Scalar(128, 0, 128);
                    cv::imshow("Glitch", result);
                }
                else if (key == 'c') {
                    // Reduce colors (quantize)
                    cv::Mat quantized;
                    int levels = 8;
                    blurQuantize(frame, quantized, levels);
                    
                    // Get edges
                    cv::Mat sobelX, sobelY, edges;
                    sobelX3x3(frame, sobelX);
                    sobelY3x3(frame, sobelY);
                    magnitude(sobelX, sobelY, edges);

                    // Convert edges to grayscale first
                    cv::Mat edgesGray;
                    cv::cvtColor(edges, edgesGray, cv::COLOR_BGR2GRAY);
                    
                    // Threshold edges to make them bold black lines
                    cv::Mat edgeMask;
                    cv::threshold(edgesGray, edgeMask, 30, 255, cv::THRESH_BINARY_INV);
                    
                    // Convert edge mask to 3 channels
                    cv::Mat edgeMask3;
                    cv::cvtColor(edgeMask, edgeMask3, cv::COLOR_GRAY2BGR);
                    
                    // Combine - multiply to overlay black edges
                    cv::Mat cartoon;
                    cv::bitwise_and(quantized, edgeMask3, cartoon);
                    
                    cv::imshow("Cartoon", cartoon);
                }
                else if (key == 'a') {
                    float scale_factor = 512.0 / (frame.rows > frame.cols ? frame.cols : frame.rows);
                    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

                    da_net.set_input(frame, scale_factor);

                    cv::Mat depthFrame;
                    da_net.run_network(depthFrame, frame.size());

                    // Apply bilateral filter to smooth the depth image while preserving edges
                    cv::Mat depthFiltered;
                    cv::bilateralFilter(depthFrame, depthFiltered, 9, 75, 75);
                    cv::normalize(depthFiltered, depthFiltered, 0, 255, cv::NORM_MINMAX, CV_8U);

                    cv::Mat depthVis;
                    cv::applyColorMap(depthFiltered, depthVis, cv::COLORMAP_INFERNO);

                    // Add face detection on top of depth visualization
                    cv::Mat greyFrame;
                    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                    std::vector<cv::Rect> faces;
                    detectFaces(greyFrame, faces);
                    

                    // Draw sparkles on each face
                    for (size_t i = 0; i < faces.size(); i++) {
                        int numSparkles = 30; // Number of sparkles per face
                        
                        for (int j = 0; j < numSparkles; j++) {
                            // Random position within face region
                            int x = faces[i].x + rand() % faces[i].width;
                            int y = faces[i].y + rand() % faces[i].height;
                            
                            // Random sparkle size
                            int size = rand() % 5 + 2;
                            
                            // Random bright color (white, yellow, cyan)
                            cv::Scalar colors[] = {
                                cv::Scalar(255, 255, 255),  // White
                                cv::Scalar(0, 255, 255),    // Yellow
                                cv::Scalar(255, 255, 0),    // Cyan
                                cv::Scalar(255, 200, 255)   // Light pink
                            };
                            cv::Scalar color = colors[rand() % 4];
                            
                            // Draw a 4-pointed star sparkle
                            cv::line(depthVis, cv::Point(x - size, y), cv::Point(x + size, y), color, 1);
                            cv::line(depthVis, cv::Point(x, y - size), cv::Point(x, y + size), color, 1);
                            cv::line(depthVis, cv::Point(x - size/2, y - size/2), cv::Point(x + size/2, y + size/2), color, 1);
                            cv::line(depthVis, cv::Point(x - size/2, y + size/2), cv::Point(x + size/2, y - size/2), color, 1);
                        }
                    }

                    cv::imshow("Depth Image with Faces", depthVis);
                }
                else if (key == 'k') {
                    background = frame.clone();
                    std::cout << "Background captured!" << std::endl;
                }
                else if (key == 'l') {
                    cloakMode = !cloakMode;
                    std::cout << "Body cloak: " << (cloakMode ? "ON" : "OFF") << std::endl;
                }
        }

        delete capdev;
        return(0);
}

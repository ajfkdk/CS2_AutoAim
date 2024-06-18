#ifndef CAPTURE_PROCESS_H
#define CAPTURE_PROCESS_H

#include <windows.h>
#include <opencv2/opencv.hpp>

// Capture a region of the screen
cv::Mat capture_region(int left, int top, int right, int bottom);

// Get the coordinates of the center region of the screen
RECT get_center_region();

// Capture the center of the screen
cv::Mat capture_center_screen();

#endif // CAPTURE_PROCESS_H
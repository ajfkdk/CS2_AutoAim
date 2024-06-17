#pragma once

#include <opencv2/opencv.hpp>
#include "inference.h"

class AIInferenceModule {
public:
    AIInferenceModule();
    ~AIInferenceModule();
    std::pair<cv::Mat, std::vector<DL_RESULT>> processImage(const cv::Mat& image);

private:
    YOLO_V8* yoloDetector;
    void initializeDetector();
};
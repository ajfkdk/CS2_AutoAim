#pragma once

#include <opencv2/opencv.hpp>
#include "inference.h"

class AIInferenceModule {
public:
    AIInferenceModule();
    ~AIInferenceModule();
    std::vector<DL_RESULT> AIInferenceModule::processImage(const cv::Mat& image);

private:
    YOLO_V8* yoloDetector;
    void initializeDetector();
};
// main.cpp
#include "AIInferenceModule.h"

// π”√USE_CUDA∫Í

int main() {
    AIInferenceModule aiInferenceModule;

    auto imgPath = "C:/Users/pc/PycharmProjects/pythonProject/img.png";
    cv::Mat img = cv::imread(imgPath);

    if (img.empty()) {
        std::cerr << "Could not read the image: " << imgPath << std::endl;
        return 1;
    }

    auto result = aiInferenceModule.processImage(img);
    cv::Mat globalProcessedImage = result.first;
    std::vector<DL_RESULT> globalPositionData = result.second;

    cv::imshow("output", globalProcessedImage);
    cv::waitKey(0);

    return 0;
}
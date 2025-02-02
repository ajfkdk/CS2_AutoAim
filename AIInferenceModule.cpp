#include "AIInferenceModule.h"
#include "global_config.h"
#include <iostream>
#include <filesystem> // C++17及以上版本
#define CUDA

AIInferenceModule::AIInferenceModule() {
    yoloDetector = new YOLO_V8();
    initializeDetector();
}

AIInferenceModule::~AIInferenceModule() {
    delete yoloDetector;
}

void AIInferenceModule::initializeDetector() {
    yoloDetector->classes = { "ct_body", "ct_head","t_body","t_head"};

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.modelPath = model_path;
    params.imgSize = { INPUT_SIZE, INPUT_SIZE };

    // 检查文件是否存在
    if (std::filesystem::exists(model_path)) {
        std::cout << "Model file exists." << std::endl;
    }
    else {
        std::cerr << "Model file does not exist. please check!" << std::endl;
        return;
    }

#ifdef USE_CUDA
    params.cudaEnable = true;
    //params.modelType = YOLO_DETECT_V10;
    params.modelType = YOLO_DETECT_V8;
#else
    params.cudaEnable = false;
    params.modelType = YOLO_DETECT_V8;
#endif
    yoloDetector->CreateSession(params);
}

std::vector<DL_RESULT> AIInferenceModule::processImage(const cv::Mat& image) {
    cv::Mat outputImage = image.clone();
    std::vector<DL_RESULT> results;

    yoloDetector->RunSession(outputImage, results);

    if (show_image)
    {
        // 设置窗口名称
        const std::string windowName = "processed image";


        // 绘制检测结果
        for (auto& re : results) {
            cv::Scalar color(156, 219, 250);
            cv::rectangle(outputImage, re.box, color, 3);
            float confidence = floor(100 * re.confidence) / 100;
            std::string label = yoloDetector->classes[re.classId] + " " +
                std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
            cv::rectangle(
                outputImage,
                cv::Point(re.box.x, re.box.y - 25),
                cv::Point(re.box.x + label.length() * 15, re.box.y),
                color,
                cv::FILLED
            );
            cv::putText(
                outputImage,
                label,
                cv::Point(re.box.x, re.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 0, 0),
                2
            );
        }
        // 设置窗口置顶
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::setWindowProperty(windowName, cv::WND_PROP_TOPMOST, 1);
        // 逻辑模块显示图像并处理位置数据为业务数据
        cv::imshow(windowName, outputImage);
        cv::waitKey(1);
    }
    else {
        cv::destroyAllWindows();
    }

    return results;
}
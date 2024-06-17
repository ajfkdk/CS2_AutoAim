#include "ScreanshotModule.h"
#include "AIInferenceModule.h"
#include <thread>
#include <chrono>
// 全局变量用于线程间数据传递
cv::Mat globalImageData;
cv::Mat globalProcessedImage;
std::vector<DL_RESULT> globalPositionData;
// 标志程序运行状态
bool running = true;

void screenshotThread() {
    while (running) {
        std::string window_name = "Counter-Strike 2"; // 替换为你想要截图的窗口标题
        globalImageData = capture_window_center_region(window_name,320,320);
        
    }
}

// AI推理线程函数
void aiInferenceThread(AIInferenceModule& aiInferenceModule) {
    while (running) {
        if (!globalImageData.empty()) {
            auto result = aiInferenceModule.processImage(globalImageData);
            globalProcessedImage = result.first;
            globalPositionData = result.second;
            // 处理完后清空imageData
            globalImageData.release();
        }
    }
}


int main() {
    // 创建模块实例
    AIInferenceModule aiInferenceModule;
    // 启动截图和AI推理线程
    std::thread screenshot(screenshotThread);
    std::thread aiInference(aiInferenceThread, std::ref(aiInferenceModule));

    // 设置窗口名称
    const std::string windowName = "processed image";

    while (true) {
        if (!globalProcessedImage.empty()) {
            // 设置窗口置顶
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::setWindowProperty(windowName, cv::WND_PROP_TOPMOST, 1);
            // 逻辑模块显示图像并处理位置数据为业务数据
            cv::imshow(windowName, globalProcessedImage);
            cv::waitKey(1);

            // 清空已处理的数据
            globalProcessedImage.release();
            globalPositionData.clear();
        }
    }

    // 等待线程结束
    screenshot.join();
    aiInference.join();

    return 0;
}
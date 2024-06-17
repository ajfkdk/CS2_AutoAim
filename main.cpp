#include "ScreanshotModule.h"
#include "AIInferenceModule.h"
#include <thread>
#include <chrono>
// ȫ�ֱ��������̼߳����ݴ���
cv::Mat globalImageData;
cv::Mat globalProcessedImage;
std::vector<DL_RESULT> globalPositionData;
// ��־��������״̬
bool running = true;

void screenshotThread() {
    while (running) {
        std::string window_name = "Counter-Strike 2"; // �滻Ϊ����Ҫ��ͼ�Ĵ��ڱ���
        globalImageData = capture_window_center_region(window_name,320,320);
        
    }
}

// AI�����̺߳���
void aiInferenceThread(AIInferenceModule& aiInferenceModule) {
    while (running) {
        if (!globalImageData.empty()) {
            auto result = aiInferenceModule.processImage(globalImageData);
            globalProcessedImage = result.first;
            globalPositionData = result.second;
            // ����������imageData
            globalImageData.release();
        }
    }
}


int main() {
    // ����ģ��ʵ��
    AIInferenceModule aiInferenceModule;
    // ������ͼ��AI�����߳�
    std::thread screenshot(screenshotThread);
    std::thread aiInference(aiInferenceThread, std::ref(aiInferenceModule));

    // ���ô�������
    const std::string windowName = "processed image";

    while (true) {
        if (!globalProcessedImage.empty()) {
            // ���ô����ö�
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::setWindowProperty(windowName, cv::WND_PROP_TOPMOST, 1);
            // �߼�ģ����ʾͼ�񲢴���λ������Ϊҵ������
            cv::imshow(windowName, globalProcessedImage);
            cv::waitKey(1);

            // ����Ѵ��������
            globalProcessedImage.release();
            globalPositionData.clear();
        }
    }

    // �ȴ��߳̽���
    screenshot.join();
    aiInference.join();

    return 0;
}
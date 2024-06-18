#include <winsock2.h>
#include <windows.h>
#include "ScreanshotModule.h"
#include "AIInferenceModule.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include "UDPSender.h"

cv::Mat globalImageData;
cv::Mat globalProcessedImage;
std::vector<DL_RESULT> globalPositionData;
bool running = true;
// 用于交换读写缓冲区
std::vector<DL_RESULT> buffer1;
std::vector<DL_RESULT> buffer2;
std::vector<DL_RESULT>* writeBuffer = &buffer1;
std::vector<DL_RESULT>* readBuffer = &buffer2;
// 用于通知新数据可用
std::atomic<bool> newDataAvailable(false);

const int screen_width = 1920;
const int screen_height = 1080;
const int CAPTURE_SIZE = 320;
const int image_top_left_x = (screen_width - CAPTURE_SIZE) / 2;
const int image_top_left_y = (screen_height - CAPTURE_SIZE) / 2;

// 自瞄强度，可以根据需要调整
float aim_strength = 0.5; 

// 截图线程
void screenshotThread() {
    while (running) {
        globalImageData = capture_center_screen();
    }
}

// AI 推理线程
void aiInferenceThread(AIInferenceModule& aiInferenceModule) {
    while (running) {
        if (!globalImageData.empty()) {
            auto results = aiInferenceModule.processImage(globalImageData);
            writeBuffer->assign(results.begin(), results.end());
            newDataAvailable.store(true, std::memory_order_release);
            std::swap(writeBuffer, readBuffer);
        }
    }
}

// 寻找并计算移动向量
std::pair<int, int> find_and_calculate_vector(const std::vector<DL_RESULT>& boxes, float aim_strength, float target_adjustment = 0.5) {
    int target_x = CAPTURE_SIZE / 2;
    int target_y = CAPTURE_SIZE / 2;
    float min_distance = std::numeric_limits<float>::infinity();
    int nearest_bbox_x = target_x;
    int nearest_bbox_y = target_y;

    for (const auto& box : boxes) {
        int bbox_x = (box.box.x + box.box.x + box.box.width) / 2;
        int bbox_y = (box.box.y + box.box.y + box.box.height) / 2;
        int adjusted_bbox_y = bbox_y - static_cast<int>((box.box.height) * target_adjustment);
        float distance = std::pow(bbox_x - target_x, 2) + std::pow(adjusted_bbox_y - target_y, 2);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_bbox_x = bbox_x;
            nearest_bbox_y = adjusted_bbox_y;
        }
    }

    nearest_bbox_x += image_top_left_x;
    nearest_bbox_y += image_top_left_y;

    int center_x = screen_width / 2;
    int center_y = screen_height / 2;
    int dx = nearest_bbox_x - center_x;
    int dy = nearest_bbox_y - center_y;

    float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
    if (distance == 0) {
        return { 0, 0 };
    }

    int max_step = 20;
    int min_step = 2;
    float step = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance / 10.0f));

    step *= aim_strength;

    float direction_x = dx / distance;
    float direction_y = dy / distance;
    int move_x = static_cast<int>(direction_x * step);
    int move_y = static_cast<int>(direction_y * step);

    return { move_x, move_y };
}



int main() {
    AIInferenceModule aiInferenceModule;
    std::thread screenshot(screenshotThread);
    std::thread aiInference(aiInferenceThread, std::ref(aiInferenceModule));

    // 配置UDP发送器
    std::string udp_ip = "192.168.8.7"; // 目标设备的IP地址
    unsigned short udp_port = 12345; // 目标设备的端口

    UDPSender udpSender(udp_ip, udp_port);
    udpSender.start();
    while (true) {
        if (newDataAvailable.load(std::memory_order_acquire)) {
            newDataAvailable.store(false, std::memory_order_release);

            if (readBuffer && !readBuffer->empty()) {
                for (const auto& position : *readBuffer) {
                    std::cout << "class: " << position.classId << " ";
                    std::cout << "confidence: " << position.confidence << " ";
                    std::cout << "position: " << position.box << std::endl;
                }

                float aim_strength = 2; // 自瞄强度，可以根据需要调整
                auto [move_x, move_y] = find_and_calculate_vector(*readBuffer, aim_strength,0.1);

                std::cout << "Move vector: (" << move_x << ", " << move_y << ")" << std::endl;

                udpSender.updatePosition(move_x, move_y);
            }
        }
    }
    //等待10秒
    std::this_thread::sleep_for(std::chrono::seconds(10));
    screenshot.join();
    aiInference.join();
    udpSender.stop();

    return 0;
}
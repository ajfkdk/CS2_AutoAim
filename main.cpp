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
#include "MouseController.h"
#include "ThreadPool.h"

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
// 用于记录侧键状态
std::atomic<bool> isXButton1Pressed{ false };
std::atomic<bool> isXButton2Pressed{ false };
// 连续发射子弹数量
int bullet_count = 1;

//推理图片的双缓冲
cv::Mat bufferImage1;
cv::Mat bufferImage2;
cv::Mat* writeImageBuffer = &bufferImage1;
cv::Mat* readImageBuffer = &bufferImage2;
std::atomic<bool> imageBufferReady(false);

// 发射状态
std::atomic<bool> isFiring(false);
// 上一次点射结束时间
std::chrono::time_point<std::chrono::steady_clock> lastBurstEndTime;

const int screen_width = 1920;
const int screen_height = 1080;
const int CAPTURE_SIZE = 320;
const int image_top_left_x = (screen_width - CAPTURE_SIZE) / 2;
const int image_top_left_y = (screen_height - CAPTURE_SIZE) / 2;

// 创建一个全局的线程池对象
ThreadPool pool(std::thread::hardware_concurrency());

//声明一个鼠标钩子
HHOOK mouseHook;
LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam);
// 配置UDP发送器
std::string udp_ip = "192.168.8.7"; // 目标设备的IP地址
unsigned short udp_port = 12345; // 目标设备的端口

// 将 udpSender 定义为全局变量
UDPSender udpSender(udp_ip, udp_port);
// 自瞄强度，可以根据需要调整
float aim_strength = 1;

// 截图线程
void screenshotThread() {
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);

    while (running) {
        cv::Mat newImage = capture_center_screen();
        *writeImageBuffer = newImage;
        imageBufferReady.store(true, std::memory_order_release);
        std::swap(writeImageBuffer, readImageBuffer);
    }
}

// AI 推理线程
void aiInferenceThread(AIInferenceModule& aiInferenceModule) {
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);

    while (running) {
        if (imageBufferReady.load(std::memory_order_acquire)) {
            cv::Mat imageToProcess = readImageBuffer->clone();
            imageBufferReady.store(false, std::memory_order_release);

            if (!imageToProcess.empty()) {
                auto results = aiInferenceModule.processImage(imageToProcess);
                writeBuffer->assign(results.begin(), results.end());
                newDataAvailable.store(true, std::memory_order_release);
                std::swap(writeBuffer, readBuffer);
            }
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

    int max_step = 10;
    int min_step = 1;
    float step = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance / 10.0f));

    step *= aim_strength;

    float direction_x = dx / distance;
    float direction_y = dy / distance;
    int move_x = static_cast<int>(direction_x * step);
    int move_y = static_cast<int>(direction_y * step);

    return { move_x, move_y };
}

int find_and_calculate_vector_x(const std::vector<DL_RESULT>& boxes, float aim_strength) {
    // 定义捕获区域中心的目标位置
    int target_x = CAPTURE_SIZE / 2;

    // 初始化最小距离为一个非常大的值
    float min_distance = std::numeric_limits<float>::infinity();

    // 用于存储最近边界框中心的变量
    int nearest_bbox_x = target_x;

    // 遍历所有边界框以找到最近的一个
    for (const auto& box : boxes) {
        // 计算边界框的中心
        int bbox_x = (box.box.x + box.box.x + box.box.width) / 2;

        // 计算目标位置的距离
        float distance = std::pow(bbox_x - target_x, 2); // 只计算x方向的距离

        // 更新最近边界框中心
        if (distance < min_distance) {
            min_distance = distance;
            nearest_bbox_x = bbox_x;
        }
    }

    // 调整最近边界框的坐标到屏幕空间
    nearest_bbox_x += image_top_left_x;

    // 定义屏幕中心
    int center_x = screen_width / 2;

    // 计算与屏幕中心的差值
    int dx = nearest_bbox_x - center_x;

    // 如果距离为0，不需要移动
    if (dx == 0) {
        return 0;
    }

    // 定义最大和最小步长
    int max_step = 10;
    int min_step = 1;

    // 计算步长
    float distance = std::abs(dx);
    float step = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance / 10.0f));

    // 调整步长基于瞄准强度
    step *= aim_strength;

    // 计算移动量
    int move_x = static_cast<int>(dx / std::abs(dx) * step);

    // 返回x方向的移动向量
    return move_x;
}

void burstFire() {
    for (int i = 0; i < bullet_count; ++i) {
        std::cout<<"===>"<<std::endl;
        MouseController::click();
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // 每发子弹之间稍微停顿
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(300)); // 点射结束之后的停顿
    isFiring.store(false, std::memory_order_release);
}

bool isAllowedMove(int move_x, int move_y) {
     // 判断move_x是否在-1到1之间
		return (move_x >= -1 && move_x <= 1);
}

// 线程函数
void processXButton1() {
    while (isXButton1Pressed.load(std::memory_order_acquire)) {
        if (newDataAvailable.load(std::memory_order_acquire)) {
            newDataAvailable.store(false, std::memory_order_release);
            if (readBuffer && !readBuffer->empty()) {
                int move_x = find_and_calculate_vector_x(*readBuffer, 2);
                udpSender.updatePosition(move_x, 0);
                if (isAllowedMove(move_x,0)) {

                    // 如果没有在点射中并且距离上一次点射结束已经超过 500 毫秒，则启动新的点射
                    if (!isFiring.load(std::memory_order_acquire)) {
                        isFiring.store(true, std::memory_order_release);
                        std::thread(burstFire).detach();
                    }

                }

            }
        }
    }
}

void processXButton2() {
    while (isXButton2Pressed.load(std::memory_order_acquire)) {
        if (newDataAvailable.load(std::memory_order_acquire)) {
            newDataAvailable.store(false, std::memory_order_release);
            if (readBuffer && !readBuffer->empty()) {
                int move_x = find_and_calculate_vector_x(*readBuffer, aim_strength);
                udpSender.updatePosition(move_x, 0);
            }
        }
    }
}

LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        MSLLHOOKSTRUCT* mouseInfo = (MSLLHOOKSTRUCT*)lParam;

        if (wParam == WM_XBUTTONDOWN) {
            if (HIWORD(mouseInfo->mouseData) & XBUTTON1) {
                std::cout << "侧键1按下" << std::endl;
                if (!isXButton1Pressed.exchange(true, std::memory_order_acq_rel)) {
                    pool.enqueue(processXButton1);
                }
            }
            else if (HIWORD(mouseInfo->mouseData) & XBUTTON2) {
                std::cout << "侧键2按下" << std::endl;
                if (!isXButton2Pressed.exchange(true, std::memory_order_acq_rel)) {
                    pool.enqueue(processXButton2);
                }
            }
        }
        else if (wParam == WM_XBUTTONUP) {
            if (HIWORD(mouseInfo->mouseData) & XBUTTON1) {
                std::cout << "侧键1松开" << std::endl;
                isXButton1Pressed.store(false, std::memory_order_release);
            }
            else if (HIWORD(mouseInfo->mouseData) & XBUTTON2) {
                std::cout << "侧键2松开" << std::endl;
                isXButton2Pressed.store(false, std::memory_order_release);
            }
        }
    }
    return CallNextHookEx(mouseHook, nCode, wParam, lParam);
}


void setProcessPriority() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        std::cerr << "Failed to set process priority!" << std::endl;
    }

    if (!SetProcessPriorityBoost(GetCurrentProcess(), FALSE)) {
        std::cerr << "Failed to disable priority boost!" << std::endl;
    }
}


int main() {
    // 设置进程优先级
    setProcessPriority();

    // 创建截图线程并加入线程池
    pool.enqueue(screenshotThread);

    // 创建 AI 推理模块
    AIInferenceModule aiInferenceModule;

    // 创建 AI 推理线程并加入线程池
    pool.enqueue([&aiInferenceModule] { aiInferenceThread(aiInferenceModule); });

    udpSender.start();

    // 设置鼠标钩子
    mouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, nullptr, 0);
    if (!mouseHook) {
        std::cerr << "Failed to install mouse hook!" << std::endl;
        return 1;
    }

    // 消息循环
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 卸载鼠标钩子
    UnhookWindowsHookEx(mouseHook);

    // 停止所有线程
    running = false;

    return 0;
}
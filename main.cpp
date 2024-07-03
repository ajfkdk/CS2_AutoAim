﻿#include "UDPSender.h"
#include "MouseController.h"
#include "ScreanshotModule.h"
#include "AIInferenceModule.h"
#include "GUIModule.h"
#include "ThreadPool.h"
#include "global_config.h"
#include <winsock2.h>
#include <windows.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <TCPSender.h>

bool debugAI = true;
bool debugCapture = false;

cv::Mat globalImageData;
cv::Mat globalProcessedImage;
std::vector<DL_RESULT> globalPositionData;
// 标志程序运行状态
std::atomic<bool> running(true);
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

//推理图片的双缓冲
cv::Mat bufferImage1;
cv::Mat bufferImage2;
cv::Mat* writeImageBuffer = &bufferImage1;
cv::Mat* readImageBuffer = &bufferImage2;
std::atomic<bool> imageBufferReady(false);

// 发射状态
std::atomic<bool> isFiring(false);

const int screen_width = 1920;
const int screen_height = 1080;
const int CAPTURE_SIZE = 320;
const int image_top_left_x = (screen_width - CAPTURE_SIZE) / 2;
const int image_top_left_y = (screen_height - CAPTURE_SIZE) / 2;

// 创建一个全局的线程池对象
ThreadPool pool(std::thread::hardware_concurrency());

//声明一个鼠标钩子
HHOOK mouseHook;
HHOOK keyboardHook;
LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam);

TCPSender tcpSender(tcp_ip, tcp_port);
// 将 udpSender 定义为全局变量
UDPSender udpSender(udp_ip, udp_port);


//主线程ID用于发送WM_QUIT消息
DWORD mainThreadId;

GUIModule guiModule(running);


// 截图线程
void screenshotThread() {
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);
    std::cout << "Screenshot thread running: " << running.load() << std::endl;
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat newImage = capture_center_screen();
        *writeImageBuffer = newImage;
        imageBufferReady.store(true, std::memory_order_release);
        std::swap(writeImageBuffer, readImageBuffer);

        auto end = std::chrono::high_resolution_clock::now();
        if (debugCapture){
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Screenshot time: " << elapsed.count() * 1000 << " ms" << std::endl;
        }
        //暂停1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    }
    //关闭getMessage
    PostThreadMessage(mainThreadId, WM_QUIT, 0, 0);  // 发送 WM_QUIT 消息
    std::cout << "Screenshot thread running: " << running.load() << std::endl;
}

// AI 推理线程
void aiInferenceThread(AIInferenceModule& aiInferenceModule) {
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);

    while (running) {
       
        
        if (imageBufferReady.load(std::memory_order_acquire)) {
            cv::Mat imageToProcess = readImageBuffer->clone();
            imageBufferReady.store(false, std::memory_order_release);

            if (!imageToProcess.empty()) {
                auto start = std::chrono::high_resolution_clock::now();
                auto results = aiInferenceModule.processImage(imageToProcess);
                writeBuffer->assign(results.begin(), results.end());
                newDataAvailable.store(true, std::memory_order_release);
                std::cout<<"AI process:" << newDataAvailable.load(std::memory_order_acquire) << std::endl;
                std::swap(writeBuffer, readBuffer);
                auto end = std::chrono::high_resolution_clock::now();
                if (debugAI) {
                    std::chrono::duration<double> elapsed = end - start;
                    std::cout << "---->AI time: " << elapsed.count() * 1000 << " ms" << std::endl;
                }
            }
            else {
                				std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        else {
            			std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
       
        
    }
    std::cout << "AiInference thread running: " << running.load() << std::endl;
    
   
}

// 寻找并计算移动向量
std::pair<int, int> find_and_calculate_vector(const std::vector<DL_RESULT>& boxes,float speed = 1.0f) {
    int target_x = CAPTURE_SIZE / 2;
    int target_y = CAPTURE_SIZE / 2; // 假设捕获区域是正方形的

    int min_distance_sq = std::numeric_limits<int>::max();
    int nearest_bbox_x = target_x;
    int nearest_bbox_y = target_y;
    bool is_head = false;

    for (const auto& box : boxes) {
        int bbox_x = (box.box.x + box.box.x + box.box.width) / 2;
        int bbox_y = (box.box.y + box.box.y + box.box.height) / 2;
        int distance_sq = (bbox_x - target_x) * (bbox_x - target_x) + (bbox_y - target_y) * (bbox_y - target_y); // 计算距离的平方

        if (distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            nearest_bbox_x = bbox_x;
            nearest_bbox_y = bbox_y;
            is_head = (box.classId == 1 || box.classId == 3); // 判断是否为头部
        }
    }

    nearest_bbox_x += image_top_left_x;
    nearest_bbox_y += image_top_left_y; // 假设image_top_left_y已定义

    int center_x = screen_width / 2;
    int center_y = screen_height / 2; // 假设已定义屏幕高度
    int dx = nearest_bbox_x - center_x;
    int dy = nearest_bbox_y - center_y;

    int max_step = 10;
    int min_step = 1;

    int move_x = 0;
    if (dx != 0) {
        int distance_x = std::abs(dx);
        float step_x = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance_x / 10.0f));
        step_x *= speed;
        move_x = static_cast<int>(dx / distance_x * step_x);
    }

    int move_y = 0;
    if (is_head && dy != 0) {
        int distance_y = std::abs(dy);
        float step_y = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance_y / 10.0f));
        step_y *= speed;
        move_y = static_cast<int>(dy / distance_y * step_y);
    }

    return { move_x, move_y };
}

void burstFire() {
    for (int i = 0; i < bullet_count; ++i) {
        std::cout<< "Firing bullet " << i + 1 << std::endl;
        udpSender.sendLeftClick();
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // 每发子弹之间稍微停顿
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(300)); // 点射结束之后的停顿
    isFiring.store(false, std::memory_order_release);
}


// 线程函数
void processXButton1() {
    while (isXButton1Pressed.load(std::memory_order_acquire)) {
        if (newDataAvailable.load(std::memory_order_acquire)) {
            newDataAvailable.store(false, std::memory_order_release);
            if (readBuffer && !readBuffer->empty()) {
                std::pair<int, int> movement = find_and_calculate_vector(*readBuffer,aim_strength2);
                int move_x = movement.first;
                int move_y = movement.second;
                udpSender.updatePosition(move_x, move_y);
                if (move_x >= -shoot_range && move_x <= shoot_range) {
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
                std::pair<int, int> movement = find_and_calculate_vector(*readBuffer,aim_strength);
                int move_x = movement.first;
                int move_y = movement.second;
                udpSender.updatePosition(move_x, move_y);
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
LRESULT CALLBACK KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0 && (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN)) {
        KBDLLHOOKSTRUCT* kbdStruct = (KBDLLHOOKSTRUCT*)lParam;
        switch (kbdStruct->vkCode) {
        case VK_PRIOR: // PageUp key
            guiModule.showWindow();
            break;
        case VK_NEXT: // PageDown key
            guiModule.hideWindow();
            break;
        }
    }
    return CallNextHookEx(keyboardHook, nCode, wParam, lParam);
}

void setProcessPriority() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        std::cerr << "Failed to set process priority!" << std::endl;
    }

    if (!SetProcessPriorityBoost(GetCurrentProcess(), FALSE)) {
        std::cerr << "Failed to disable priority boost!" << std::endl;
    }
}

void setHooks() {
    mouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, nullptr, 0);
    if (!mouseHook) {
        std::cerr << "Failed to install mouse hook!" << std::endl;
    }

    keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProc, nullptr, 0);
    if (!keyboardHook) {
        std::cerr << "Failed to install keyboard hook!" << std::endl;
    }
}

void removeHooks() {
    if (mouseHook) {
        UnhookWindowsHookEx(mouseHook);
    }
    if (keyboardHook) {
        UnhookWindowsHookEx(keyboardHook);
    }
}

//int  WINAPI WinMain(HINSTANCE hInstance,
//    HINSTANCE hPrevInstance,
//    LPSTR lpCmdLine,
//    int nCmdShow) {
//    // 获取主线程 ID
//    mainThreadId = GetCurrentThreadId();
//    // 设置进程优先级
//    setProcessPriority();
//
//    // 创建截图线程并加入线程池
//    pool.enqueue(screenshotThread);
//
//    // 创建 AI 推理模块
//    AIInferenceModule aiInferenceModule;
//
//    // 创建 AI 推理线程并加入线程池
//    pool.enqueue([&aiInferenceModule] { aiInferenceThread(aiInferenceModule); });
//
//
//    //guiModule.start();
//    udpSender.start();
//    guiModule.start();
//
//    // 设置键鼠钩子
//    setHooks();
//    guiModule.hideWindow();
//    // 消息循环
//    MSG msg;
//    while (running && GetMessage(&msg, nullptr, 0, 0)) {
//        std::cout<< "111 Main thread running: " << running.load() << std::endl;
//        TranslateMessage(&msg);
//        DispatchMessage(&msg);
//    }
//    std::cout << "Main thread running: " << running.load() << std::endl;
//
//    // 移除钩子
//    removeHooks();
//
//    // 停止所有线程
//    running = false;
//
//    // 等待线程池中的所有任务完成
//    pool.~ThreadPool();
//
//    return 0;
//}
//
void checkCUDA() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // Get CUDA driver version
    int driverVersion = 0;
    cudaStatus = cudaDriverGetVersion(&driverVersion);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get CUDA driver version! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // Get CUDA runtime version
    int runtimeVersion = 0;
    cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get CUDA runtime version! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // Get the number of CUDA devices
    int deviceCount = 0;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
    }
    else {
        std::cout << "CUDA is available. Number of CUDA devices: " << deviceCount << std::endl;
        std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
        std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
    }
}

void checkONNXRuntime() {
    // Get ONNX Runtime version
    auto version = Ort::GetVersionString();
    std::cout << "ONNX Runtime version: " << version << std::endl;
}


int  main123() {
    checkCUDA();
    //checkONNXRuntime();
    // 获取主线程 ID
    mainThreadId = GetCurrentThreadId();
    // 设置进程优先级
    setProcessPriority();

    //// 创建截图线程并加入线程池
    //pool.enqueue(screenshotThread);


    // 创建 AI 推理模块
    AIInferenceModule aiInferenceModule;

    // 创建 AI 推理线程并加入线程池
    pool.enqueue([&aiInferenceModule] { aiInferenceThread(aiInferenceModule); });


    //guiModule.start();
    udpSender.start();
    guiModule.start();

    // 设置键鼠钩子
    setHooks();
    // 消息循环
    MSG msg;
    while (running && GetMessage(&msg, nullptr, 0, 0)) {
        std::cout << "111 Main thread running: " << running.load() << std::endl;
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    std::cout << "Main thread running: " << running.load() << std::endl;

    // 移除钩子
    removeHooks();

    // 停止所有线程
    running = false;

    // 等待线程池中的所有任务完成
    pool.~ThreadPool();

    return 0;
}

int  main231() {
    // 获取主线程 ID
    mainThreadId = GetCurrentThreadId();
    // 设置进程优先级
    setProcessPriority();

    tcpSender.setImageBuffers(&bufferImage1, &bufferImage2, &imageBufferReady);

    std::thread tcpSenderThread([&]() {
        tcpSender.start();
     });

    // 创建 TCP 图像接收线程并加入线程池


    // 创建 AI 推理模块
    AIInferenceModule aiInferenceModule;

    // 创建 AI 推理线程并加入线程池
    pool.enqueue([&aiInferenceModule] { aiInferenceThread(aiInferenceModule); });


    //guiModule.start();
    udpSender.start();
    guiModule.start();

    // 设置键鼠钩子
    setHooks();
    guiModule.hideWindow();
    // 消息循环
    MSG msg;
    while (running && GetMessage(&msg, nullptr, 0, 0)) {
        std::cout << "111 Main thread running: " << running.load() << std::endl;
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    std::cout << "Main thread running: " << running.load() << std::endl;

    // 移除钩子
    removeHooks();

    // 停止所有线程
    running = false;

    tcpSender.stop();
    if (tcpSenderThread.joinable()) {
        tcpSenderThread.join();
    }

    // 等待线程池中的所有任务完成
    pool.~ThreadPool();

    return 0;
}




int main12() {
    // 创建 AI 推理模块
    AIInferenceModule aiInferenceModule;

    // 读取图片
    std::string imagePath = "C:/Users/pc/Desktop/Snipaste_2024-06-17_21-35-30.png";
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // 进行 AI 推理
    auto results = aiInferenceModule.processImage(image);

    // 输出推理结果
    for (const auto& result : results) {
        std::cout << "Detection: " << result.box.x << ", " << result.box.y << ", " << result.box.width << ", " << result.box.height << std::endl;
    }

    return 0;
}


const std::string SERVER_IP = "192.168.8.6";  // 替换为你的服务器IP地址
const unsigned short SERVER_PORT = 18856;

void send_image(boost::asio::ip::tcp::socket& socket) {
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat newImage = capture_center_screen();
        std::vector<uchar> img_bytes;
        cv::imencode(".jpg", newImage, img_bytes, { cv::IMWRITE_JPEG_QUALITY, 85 });

        // 发送图像长度和图像数据
        try {
            uint32_t img_size = htonl(img_bytes.size());
            boost::asio::write(socket, boost::asio::buffer(&img_size, sizeof(img_size)));
            boost::asio::write(socket, boost::asio::buffer(img_bytes));
        }
        catch (std::exception& e) {
            std::cerr << "连接中断: " << e.what() << std::endl;
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        if (debugCapture) {
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Screenshot time: " << elapsed.count() * 1000 << " ms" << std::endl;
        }
     

    }
}

const std::string SERVER_IP = "192.168.8.6";  // 替换为你的服务器IP地址
const unsigned short SERVER_PORT = 18856;     // 替换为你的服务器端口
std::atomic<bool> running(true);
bool debugCapture = true;
boost::asio::ip::tcp::socket* global_socket = nullptr; // 全局socket指针


void send_image(boost::asio::ip::tcp::socket& socket) {
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat newImage = capture_center_screen();
        if (newImage.empty()) {
            std::cerr << "Failed to capture center screen." << std::endl;
            continue;
        }

        std::vector<uchar> img_bytes;
        cv::imencode(".jpg", newImage, img_bytes, { cv::IMWRITE_JPEG_QUALITY, 85 });

        try {
            uint32_t protocol = htonl(3);  // 3 表示图片协议头
            boost::asio::write(socket, boost::asio::buffer(&protocol, sizeof(protocol)));

            uint32_t img_size = htonl(img_bytes.size());
            boost::asio::write(socket, boost::asio::buffer(&img_size, sizeof(img_size)));
            boost::asio::write(socket, boost::asio::buffer(img_bytes));
        }
        catch (std::exception& e) {
            std::cerr << "连接中断: " << e.what() << std::endl;
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        if (debugCapture) {
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Screenshot time: " << elapsed.count() * 1000 << " ms" << std::endl;
        }

        // 暂停5ms
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

LRESULT CALLBACK MouseHookProc234(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode == HC_ACTION) {
        if (wParam == WM_XBUTTONDOWN) {
            MSLLHOOKSTRUCT* pMouseStruct = (MSLLHOOKSTRUCT*)lParam;
            if (pMouseStruct != nullptr) {
                if (HIWORD(pMouseStruct->mouseData) == XBUTTON1) {
                    std::cout << "XBUTTON1 pressed" << std::endl;
                    if (global_socket) {
                        uint32_t protocol = htonl(1);  // 1 表示 MOUSE_XBOX1 协议头
                        boost::asio::write(*global_socket, boost::asio::buffer(&protocol, sizeof(protocol)));
                    }
                }
                else if (HIWORD(pMouseStruct->mouseData) == XBUTTON2) {
                    std::cout << "XBUTTON2 pressed" << std::endl;
                    if (global_socket) {
                        uint32_t protocol = htonl(2);  // 2 表示 MOUSE_BOX2 协议头
                        boost::asio::write(*global_socket, boost::asio::buffer(&protocol, sizeof(protocol)));
                    }
                }
            }
        }
    }
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}


void main_thread() {
    try {
        boost::asio::io_context io_context;
        boost::asio::ip::tcp::socket socket(io_context);
        boost::asio::ip::tcp::resolver resolver(io_context);
        boost::asio::connect(socket, resolver.resolve(SERVER_IP, std::to_string(SERVER_PORT)));

        // 将全局socket指针指向当前socket
        global_socket = &socket;

        // 安装鼠标钩子
        HHOOK mouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc234, NULL, 0);
        if (!mouseHook) {
            std::cerr << "Failed to install mouse hook!" << std::endl;
            return;
        }

        // 创建并启动发送图像线程
        std::thread send_thread(send_image, std::ref(socket));

        // 保持主线程运行以处理钩子回调
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        running = false;
        send_thread.join();

        UnhookWindowsHookEx(mouseHook);
    }
    catch (std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
    }
}

int main() {
    try {
        main_thread();
    }
    catch (std::exception& e) {
        std::cerr << "未捕获的异常: " << e.what() << std::endl;
    }

    return 0;
}
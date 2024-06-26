#include "UDPSender.h"
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
#include "mouse_logic.h"  

bool debugAI = false;
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

// 智能鼠标模块
MouseLogic mouse_logic(CAPTURE_SIZE, screen_width, screen_height, image_top_left_x , image_top_left_y);

// 创建一个全局的线程池对象
ThreadPool pool(std::thread::hardware_concurrency());

//声明一个鼠标钩子
HHOOK mouseHook;
HHOOK keyboardHook;
LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam);


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
                mouse_logic.set_boxes(results);
                
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
        auto [move_x, move_y] = mouse_logic.get_move_vector(aim_strength2);
        udpSender.updatePosition(move_x, move_y);
        std::this_thread::sleep_for(std::chrono::milliseconds(mouse_move_pause));
        if (move_x!=0&&move_x >= -shoot_range && move_x <= shoot_range) {
            if (!isFiring.load(std::memory_order_acquire)) {
                isFiring.store(true, std::memory_order_release);
                std::thread(burstFire).detach();
            }
        }
        
    }
}

void processXButton2() {
    while (isXButton2Pressed.load(std::memory_order_acquire)) {
        auto [move_x, move_y] = mouse_logic.get_move_vector(aim_strength2);
        udpSender.updatePosition(move_x, move_y);
        std::this_thread::sleep_for(std::chrono::milliseconds(mouse_move_pause));
        
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
                mouse_logic.reset_end();
            }
            else if (HIWORD(mouseInfo->mouseData) & XBUTTON2) {
                std::cout << "侧键2松开" << std::endl;
                isXButton2Pressed.store(false, std::memory_order_release);
                mouse_logic.reset_end();
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
int  main() {
    // 获取主线程 ID
    mainThreadId = GetCurrentThreadId();
    // 设置进程优先级
    setProcessPriority();


    // 创建截图线程并加入线程池
    pool.enqueue(screenshotThread);

    // 创建 AI 推理模块
    AIInferenceModule aiInferenceModule;

    // 创建 AI 推理线程并加入线程池
    pool.enqueue([&aiInferenceModule] { aiInferenceThread(aiInferenceModule); });


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


int main12() {
    // 创建 AI 推理模块
    AIInferenceModule aiInferenceModule;

    // 读取图片
    std::string imagePath = "C:/Users/pc/Desktop/Snipaste_2024-06-17_21-35-30.jpg";
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
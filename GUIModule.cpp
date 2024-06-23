#include "GUIModule.h"
#include "global_config.h"  // 包含全局配置头文件
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <thread>
#include <windows.h>

GUIModule::GUIModule(std::atomic<bool>& running) : running(running), window(nullptr) {}

void GUIModule::start() {
    std::thread(&GUIModule::guiThread, this).detach();
}

void GUIModule::stop() {
    running.store(false, std::memory_order_release);
}

void GUIModule::showWindow() {
    if (window) {
        glfwShowWindow(window);
    }
}

void GUIModule::hideWindow() {
    if (window) {
        glfwHideWindow(window);
    }
}

void GUIModule::guiThread() {
    // 初始化 GLFW
    if (!glfwInit()) {
        return;
    }

    // 创建无边框窗口
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE); // 设置窗口背景透明
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    if (!primaryMonitor) {
        glfwTerminate();
        return;
    }
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
    int screenWidth = mode->width;
    int screenHeight = mode->height;

    // 创建窗口
    window = glfwCreateWindow(screenWidth - 20, screenHeight - 20, "Control GUI", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // 设置窗口始终位于最前面
    glfwSetWindowAttrib(window, GLFW_FLOATING, GLFW_TRUE);

    // 初始化 ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // 启用 OpenGL 混合功能
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 主循环
    while (!glfwWindowShouldClose(window) && running) {
        // 开始新的 ImGui 帧
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 获取窗口大小
        int window_width, window_height;
        glfwGetWindowSize(window, &window_width, &window_height);

        // 设置 Control Panel 子窗口
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(500), static_cast<float>(500)), ImGuiCond_Once);
        ImGui::Begin("Control Panel", nullptr);

        // 添加控制参数的 ImGui 控件
        ImGui::SliderInt("Bullet Count", &bullet_count, 1, 10);
        ImGui::SliderInt("Shoot Range", &shoot_range, 1, 10);
        ImGui::InputText("UDP IP", &udp_ip[0], udp_ip.size() + 1);
        ImGui::InputScalar("UDP Port", ImGuiDataType_U16, &udp_port);
        ImGui::SliderFloat("Aim Strength", &aim_strength, 0.0f, 10.0f);
        // 临时变量来存储 std::atomic<bool> 的值
        bool show_image_temp = show_image.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Show Image", &show_image_temp)) {
            show_image.store(show_image_temp, std::memory_order_relaxed);
        }

        ImGui::InputText("Model Path", &model_path[0], model_path.size() + 1);

        if (ImGui::Button("Exit Program")) {
            exitFunction();
        }

        ImGui::End();

        // 渲染 ImGui
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        // 清空颜色缓冲区并设置透明背景
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // 交换缓冲区
        glfwSwapBuffers(window);
    }

    // 清理 ImGui 和 GLFW
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void GUIModule::exitFunction() {
    running.store(false, std::memory_order_release);
    PostMessage(nullptr, WM_QUIT, 0, 0);  // 发送 WM_QUIT 消息，确保 GetMessage 退出
}
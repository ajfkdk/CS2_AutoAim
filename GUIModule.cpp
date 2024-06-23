#include "GUIModule.h"
#include "global_config.h"  // ����ȫ������ͷ�ļ�
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
    // ��ʼ�� GLFW
    if (!glfwInit()) {
        return;
    }

    // �����ޱ߿򴰿�
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE); // ���ô��ڱ���͸��
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    if (!primaryMonitor) {
        glfwTerminate();
        return;
    }
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
    int screenWidth = mode->width;
    int screenHeight = mode->height;

    // ��������
    window = glfwCreateWindow(screenWidth - 20, screenHeight - 20, "Control GUI", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // ���ô���ʼ��λ����ǰ��
    glfwSetWindowAttrib(window, GLFW_FLOATING, GLFW_TRUE);

    // ��ʼ�� ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // ���� OpenGL ��Ϲ���
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ��ѭ��
    while (!glfwWindowShouldClose(window) && running) {
        // ��ʼ�µ� ImGui ֡
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ��ȡ���ڴ�С
        int window_width, window_height;
        glfwGetWindowSize(window, &window_width, &window_height);

        // ���� Control Panel �Ӵ���
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(500), static_cast<float>(500)), ImGuiCond_Once);
        ImGui::Begin("Control Panel", nullptr);

        // ��ӿ��Ʋ����� ImGui �ؼ�
        ImGui::SliderInt("Bullet Count", &bullet_count, 1, 10);
        ImGui::SliderInt("Shoot Range", &shoot_range, 1, 10);
        ImGui::InputText("UDP IP", &udp_ip[0], udp_ip.size() + 1);
        ImGui::InputScalar("UDP Port", ImGuiDataType_U16, &udp_port);
        ImGui::SliderFloat("Aim Strength", &aim_strength, 0.0f, 10.0f);
        // ��ʱ�������洢 std::atomic<bool> ��ֵ
        bool show_image_temp = show_image.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Show Image", &show_image_temp)) {
            show_image.store(show_image_temp, std::memory_order_relaxed);
        }

        ImGui::InputText("Model Path", &model_path[0], model_path.size() + 1);

        if (ImGui::Button("Exit Program")) {
            exitFunction();
        }

        ImGui::End();

        // ��Ⱦ ImGui
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        // �����ɫ������������͸������
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // ����������
        glfwSwapBuffers(window);
    }

    // ���� ImGui �� GLFW
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void GUIModule::exitFunction() {
    running.store(false, std::memory_order_release);
    PostMessage(nullptr, WM_QUIT, 0, 0);  // ���� WM_QUIT ��Ϣ��ȷ�� GetMessage �˳�
}
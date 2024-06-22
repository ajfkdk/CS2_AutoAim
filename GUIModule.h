#ifndef GUIMODULE_H
#define GUIMODULE_H

#include <atomic>
#include <GLFW/glfw3.h>

class GUIModule {
public:
    GUIModule(std::atomic<bool>& running);
    void start();
    void stop();
    void showWindow();
    void hideWindow();

private:
    void guiThread();
    void exitFunction();

    std::atomic<bool>& running;
    GLFWwindow* window; // 将 window 声明为成员变量
};

#endif // GUIMODULE_H
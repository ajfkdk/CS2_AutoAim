#include "MouseController.h"
#include <windows.h>
// 直接调用Windows API函数来控制鼠标时候会更加方便


// 移动鼠标的相对位置
void MouseController::moveRelative(int x, int y) {
    // 使用 MOUSEEVENTF_MOVE 标志移动鼠标
    mouse_event(MOUSEEVENTF_MOVE, x, y, 0, 0);
}

// 单击鼠标左键
void MouseController::click() {
    // 鼠标左键按下
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    // 鼠标左键抬起
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
}
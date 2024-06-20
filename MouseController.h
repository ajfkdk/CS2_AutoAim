#ifndef MOUSECONTROLLER_H
#define MOUSECONTROLLER_H

class MouseController {
public:
    // 移动鼠标的相对位置
    static void moveRelative(int x, int y);

    // 单击鼠标左键
    static void click();
};

#endif // MOUSECONTROLLER_H
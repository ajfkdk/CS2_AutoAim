#ifndef MOUSECONTROLLER_H
#define MOUSECONTROLLER_H

class MouseController {
public:
    // �ƶ��������λ��
    static void moveRelative(int x, int y);

    // ����������
    static void click();
};

#endif // MOUSECONTROLLER_H
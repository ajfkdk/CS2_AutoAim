#include "MouseController.h"
#include <windows.h>
// ֱ�ӵ���Windows API�������������ʱ�����ӷ���


// �ƶ��������λ��
void MouseController::moveRelative(int x, int y) {
    // ʹ�� MOUSEEVENTF_MOVE ��־�ƶ����
    mouse_event(MOUSEEVENTF_MOVE, x, y, 0, 0);
}

// ����������
void MouseController::click() {
    // ����������
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    // ������̧��
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
}
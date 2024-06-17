#ifndef SCREANSHOT_MODULE_H
#define SCREANSHOT_MODULE_H

#include <windows.h>
#include <string>
#include <opencv2/opencv.hpp>

// ����λͼ��Ϣͷ
BITMAPINFOHEADER create_bitmap_header(int width, int height);

// ��ȡָ������ͼ��
cv::Mat capture_region(HDC hwindowDC, int left, int top, int right, int bottom);

// ���ݴ������ƻ�ȡ���ھ��
HWND get_window_handle_by_name(const std::string& window_name);

// ��ȡ���ھ�������
RECT get_window_rect(HWND hwnd);

// ��ȡ���ڵ���������
cv::Mat capture_window_center_region(const std::string& window_name, int region_width, int region_height);

#endif // SCREANSHOT_MODULE_H
#ifndef SCREANSHOT_MODULE_H
#define SCREANSHOT_MODULE_H

#include <windows.h>
#include <string>
#include <opencv2/opencv.hpp>

// 创建位图信息头
BITMAPINFOHEADER create_bitmap_header(int width, int height);

// 截取指定区域图像
cv::Mat capture_region(HDC hwindowDC, int left, int top, int right, int bottom);

// 根据窗口名称获取窗口句柄
HWND get_window_handle_by_name(const std::string& window_name);

// 获取窗口矩形区域
RECT get_window_rect(HWND hwnd);

// 截取窗口的中心区域
cv::Mat capture_window_center_region(const std::string& window_name, int region_width, int region_height);

#endif // SCREANSHOT_MODULE_H
#include "ScreanshotModule.h"
#include <iostream>
#include <chrono>
#include <windows.h>
#include <vector>
#include <opencv2/opencv.hpp>

HDC hdesktop = GetDC(NULL);
HDC hdc = CreateCompatibleDC(hdesktop);

BITMAPINFOHEADER create_bitmap_header(int width, int height) {
    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height; // Negative height for top-down bitmap
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;
    return bi;
}

cv::Mat capture_region(HDC hwindowDC, int left, int top, int right, int bottom) {
    int width = right - left;
    int height = bottom - top;

    HBITMAP hbmp = CreateCompatibleBitmap(hwindowDC, width, height);
    SelectObject(hdc, hbmp);
    BitBlt(hdc, 0, 0, width, height, hwindowDC, left, top, SRCCOPY);

    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader = create_bitmap_header(width, height);

    std::vector<BYTE> buffer(width * height * 4);
    GetDIBits(hdc, hbmp, 0, height, buffer.data(), &bmi, DIB_RGB_COLORS);

    cv::Mat image(height, width, CV_8UC4, buffer.data());
    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

    DeleteObject(hbmp);
    return image;
}

HWND get_window_handle_by_name(const std::string& window_name) {
    return FindWindowA(NULL, window_name.c_str());
}

RECT get_window_rect(HWND hwnd) {
    RECT rect;
    GetWindowRect(hwnd, &rect);
    return rect;
}

cv::Mat capture_window_center_region(const std::string& window_name, int region_width, int region_height) {
    HWND hwnd = get_window_handle_by_name(window_name);
    if (!hwnd) {
        std::cerr << "Window not found: " << window_name << std::endl;
        return cv::Mat();
    }

    RECT rect = get_window_rect(hwnd);
    HDC hwindowDC = GetDC(hwnd);
    cv::Mat full_image = capture_region(hwindowDC, 0, 0, rect.right - rect.left, rect.bottom - rect.top);
    ReleaseDC(hwnd, hwindowDC);

    if (full_image.empty()) {
        std::cerr << "Failed to capture window: " << window_name << std::endl;
        return cv::Mat();
    }

    // Calculate the rectangle for the center region
    int center_x = (full_image.cols - region_width) / 2;
    int center_y = (full_image.rows - region_height) / 2;

    // Ensure the region is within the image bounds
    center_x = std::max(0, center_x);
    center_y = std::max(0, center_y);
    region_width = std::min(region_width, full_image.cols - center_x);
    region_height = std::min(region_height, full_image.rows - center_y);

    cv::Rect center_rect(center_x, center_y, region_width, region_height);
    return full_image(center_rect);
}

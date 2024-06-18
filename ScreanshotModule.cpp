#include "ScreanshotModule.h"
#include <iostream>
#include <chrono>

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

cv::Mat capture_region(int left, int top, int right, int bottom) {
    int width = right - left;
    int height = bottom - top;

    HBITMAP hbmp = CreateCompatibleBitmap(hdesktop, width, height);
    SelectObject(hdc, hbmp);
    BitBlt(hdc, 0, 0, width, height, hdesktop, left, top, SRCCOPY);

    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader = create_bitmap_header(width, height);

    std::vector<BYTE> buffer(width * height * 4);
    GetDIBits(hdc, hbmp, 0, height, buffer.data(), &bmi, DIB_RGB_COLORS);

    cv::Mat image(height, width, CV_8UC4, buffer.data());
    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

    DeleteObject(hbmp);
    return image;
}

RECT get_center_region() {
    int screen_width = GetSystemMetrics(SM_CXSCREEN);
    int screen_height = GetSystemMetrics(SM_CYSCREEN);
    int region_width = 320;
    int region_height = 320;

    RECT rect;
    rect.left = (screen_width - region_width) / 2;
    rect.top = (screen_height - region_height) / 2;
    rect.right = rect.left + region_width;
    rect.bottom = rect.top + region_height;

    return rect;
}

cv::Mat capture_center_screen() {
    RECT rect = get_center_region();
    return capture_region(rect.left, rect.top, rect.right, rect.bottom);
}


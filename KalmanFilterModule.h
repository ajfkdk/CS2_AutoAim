#ifndef KALMAN_FILTER_MODULE_H
#define KALMAN_FILTER_MODULE_H

#include <opencv2/opencv.hpp>
#include "global_config.h"


// 卡尔曼滤波模块
class KalmanFilterModule {
public:
    // 构造函数
    KalmanFilterModule();

    // 设置结果
    void setResult(const DL_RESULT& result);
    void reset();
    // 获取预测结果
    DL_RESULT output();

private:
    cv::KalmanFilter kalman;  // 卡尔曼滤波器
    bool initialized;         // 是否已初始化
};

#endif // KALMAN_FILTER_MODULE_H
#include "KalmanFilterModule.h"

// 构造函数
KalmanFilterModule::KalmanFilterModule() {
    // 初始化卡尔曼滤波器
    kalman = cv::KalmanFilter(6, 4);  // 6 个动态参数 (x, y, dx, dy, width, height) 和 4 个测量参数 (x, y, width, height)

    // 设置测量矩阵
    kalman.measurementMatrix = (cv::Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // 设置转移矩阵
    kalman.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // 设置过程噪声协方差矩阵
    kalman.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.03f;

    // 设置测量噪声协方差矩阵
    kalman.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);

    // 设置后验误差协方差矩阵
    kalman.errorCovPost = cv::Mat::eye(6, 6, CV_32F);

    // 初始化状态向量
    kalman.statePost = cv::Mat::zeros(6, 1, CV_32F);

    initialized = false;
}

// 设置结果
void KalmanFilterModule::setResult(const DL_RESULT& result) {
    // 提取测量值
    float x = static_cast<float>(result.box.x);
    float y = static_cast<float>(result.box.y);
    float width = static_cast<float>(result.box.width);
    float height = static_cast<float>(result.box.height);
    //std::cout << "设置值： x: " << x << " y: " << y << " width: " << width << " height: " << height << std::endl;

    // 转换为测量矩阵
    cv::Mat measurement = (cv::Mat_<float>(4, 1) << x, y, width, height);

    // 校正卡尔曼滤波器
    kalman.correct(measurement);

    // 标记卡尔曼滤波器已初始化
    initialized = true;
}

// 重置方法
void KalmanFilterModule::reset() {
    // 初始化卡尔曼滤波器
    kalman = cv::KalmanFilter(6, 4);  // 6 个动态参数 (x, y, dx, dy, width, height) 和 4 个测量参数 (x, y, width, height)

    // 设置测量矩阵
    kalman.measurementMatrix = (cv::Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // 设置转移矩阵
    kalman.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // 设置过程噪声协方差矩阵
    kalman.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.03f;

    // 设置测量噪声协方差矩阵
    kalman.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);

    // 设置后验误差协方差矩阵
    kalman.errorCovPost = cv::Mat::eye(6, 6, CV_32F);

    // 初始化状态向量
    kalman.statePost = cv::Mat::zeros(6, 1, CV_32F);

    initialized = false;
}


// 获取预测结果
DL_RESULT KalmanFilterModule::output() {
    if (!initialized) {
        DL_RESULT defaultResult;
        defaultResult.box = cv::Rect(0, 0, 50, 50);  // 默认大小为 50x50
        return defaultResult;
    }

    // 预测下一个状态
    cv::Mat prediction = kalman.predict();

    // 提取预测值
    float predictedX = prediction.at<float>(0);
    float predictedY = prediction.at<float>(1);
    float predictedWidth = prediction.at<float>(4);
    float predictedHeight = prediction.at<float>(5);
    //std::cout << "预测值： x: " << predictedX << " y: " << predictedY << " width: " << predictedWidth << " height: " << predictedHeight << std::endl;

    // 创建预测结果
    DL_RESULT predictedResult;
    predictedResult.box = cv::Rect(static_cast<int>(predictedX), static_cast<int>(predictedY),
        static_cast<int>(predictedWidth), static_cast<int>(predictedHeight));

    return predictedResult;
}
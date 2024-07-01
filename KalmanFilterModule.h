#ifndef KALMAN_FILTER_MODULE_H
#define KALMAN_FILTER_MODULE_H

#include <opencv2/opencv.hpp>
#include "global_config.h"


// �������˲�ģ��
class KalmanFilterModule {
public:
    // ���캯��
    KalmanFilterModule();

    // ���ý��
    void setResult(const DL_RESULT& result);
    void reset();
    // ��ȡԤ����
    DL_RESULT output();

private:
    cv::KalmanFilter kalman;  // �������˲���
    bool initialized;         // �Ƿ��ѳ�ʼ��
};

#endif // KALMAN_FILTER_MODULE_H
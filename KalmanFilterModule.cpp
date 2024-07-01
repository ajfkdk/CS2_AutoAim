#include "KalmanFilterModule.h"

// ���캯��
KalmanFilterModule::KalmanFilterModule() {
    // ��ʼ���������˲���
    kalman = cv::KalmanFilter(6, 4);  // 6 ����̬���� (x, y, dx, dy, width, height) �� 4 ���������� (x, y, width, height)

    // ���ò�������
    kalman.measurementMatrix = (cv::Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // ����ת�ƾ���
    kalman.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // ���ù�������Э�������
    kalman.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.03f;

    // ���ò�������Э�������
    kalman.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);

    // ���ú������Э�������
    kalman.errorCovPost = cv::Mat::eye(6, 6, CV_32F);

    // ��ʼ��״̬����
    kalman.statePost = cv::Mat::zeros(6, 1, CV_32F);

    initialized = false;
}

// ���ý��
void KalmanFilterModule::setResult(const DL_RESULT& result) {
    // ��ȡ����ֵ
    float x = static_cast<float>(result.box.x);
    float y = static_cast<float>(result.box.y);
    float width = static_cast<float>(result.box.width);
    float height = static_cast<float>(result.box.height);
    //std::cout << "����ֵ�� x: " << x << " y: " << y << " width: " << width << " height: " << height << std::endl;

    // ת��Ϊ��������
    cv::Mat measurement = (cv::Mat_<float>(4, 1) << x, y, width, height);

    // У���������˲���
    kalman.correct(measurement);

    // ��ǿ������˲����ѳ�ʼ��
    initialized = true;
}

// ���÷���
void KalmanFilterModule::reset() {
    // ��ʼ���������˲���
    kalman = cv::KalmanFilter(6, 4);  // 6 ����̬���� (x, y, dx, dy, width, height) �� 4 ���������� (x, y, width, height)

    // ���ò�������
    kalman.measurementMatrix = (cv::Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // ����ת�ƾ���
    kalman.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
    // ���ù�������Э�������
    kalman.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.03f;

    // ���ò�������Э�������
    kalman.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);

    // ���ú������Э�������
    kalman.errorCovPost = cv::Mat::eye(6, 6, CV_32F);

    // ��ʼ��״̬����
    kalman.statePost = cv::Mat::zeros(6, 1, CV_32F);

    initialized = false;
}


// ��ȡԤ����
DL_RESULT KalmanFilterModule::output() {
    if (!initialized) {
        DL_RESULT defaultResult;
        defaultResult.box = cv::Rect(0, 0, 50, 50);  // Ĭ�ϴ�СΪ 50x50
        return defaultResult;
    }

    // Ԥ����һ��״̬
    cv::Mat prediction = kalman.predict();

    // ��ȡԤ��ֵ
    float predictedX = prediction.at<float>(0);
    float predictedY = prediction.at<float>(1);
    float predictedWidth = prediction.at<float>(4);
    float predictedHeight = prediction.at<float>(5);
    //std::cout << "Ԥ��ֵ�� x: " << predictedX << " y: " << predictedY << " width: " << predictedWidth << " height: " << predictedHeight << std::endl;

    // ����Ԥ����
    DL_RESULT predictedResult;
    predictedResult.box = cv::Rect(static_cast<int>(predictedX), static_cast<int>(predictedY),
        static_cast<int>(predictedWidth), static_cast<int>(predictedHeight));

    return predictedResult;
}
#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#include <string>
#include <atomic>
#include <opencv2/opencv.hpp>
typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;

// ���������ӵ�����
inline int bullet_count = 1;

// �����Χ
inline int shoot_range = 5;

// ����UDP������
inline std::string udp_ip = "192.168.8.7"; // Ŀ���豸��IP��ַ
inline unsigned short udp_port = 21115;    // Ŀ���豸�Ķ˿�

// ����ǿ�ȣ����Ը�����Ҫ����
inline float aim_strength = 3.0f;

// ����2������ǿ��
inline float aim_strength2 = 4.0f;

// �Ƿ���ʾͼ��
inline std::atomic<bool> show_image{ true };

// ����ƶ���ͣʱ��ms
inline int mouse_move_pause = 1;

// ��С���Ŷ�
inline float min_confidence = 0.7f;

// ģ��·��
//inline std::string model_path = "C:/Users/pc/Desktop/yolov10-102.onnx";
//inline std::string model_path = "C:/Users/pc/Desktop/yolov10n2.onnx";
//inline std::string model_path = "C:/Users/pc/Desktop/yolov5.onnx";
inline std::string model_path = "C:/Users/pc/Desktop/16.onnx";
//inline std::string model_path = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx";
//inline std::string model_path = "C:/Users/pc/PycharmProjects/pythonProject/models/PUBG.onnx";

#endif // GLOBAL_CONFIG_H
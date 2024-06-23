#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#include <string>
#include <atomic>

// ���������ӵ�����
inline int bullet_count = 1;

// �����Χ
inline int shoot_range = 1;

// ����UDP������
inline std::string udp_ip = "192.168.8.7"; // Ŀ���豸��IP��ַ
inline unsigned short udp_port = 21115;    // Ŀ���豸�Ķ˿�

// ����ǿ�ȣ����Ը�����Ҫ����
inline float aim_strength = 1.0f;

// �Ƿ���ʾͼ��
inline std::atomic<bool> show_image{ true };

// ģ��·��
inline std::string model_path = "C:/Users/pc/Desktop/best.onnx";
//inline std::string model_path = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx";
//inline std::string model_path = "C:/Users/pc/PycharmProjects/pythonProject/models/PUBG.onnx";

#endif // GLOBAL_CONFIG_H
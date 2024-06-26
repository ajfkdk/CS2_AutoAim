#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#include <string>
#include <atomic>

// 连续发射子弹数量
inline int bullet_count = 1;

// 射击范围
inline int shoot_range = 5;

// 配置UDP发送器
inline std::string udp_ip = "192.168.8.7"; // 目标设备的IP地址
inline unsigned short udp_port = 21115;    // 目标设备的端口

// 自瞄强度，可以根据需要调整
inline float aim_strength = 3.0f;

// 按键2的自瞄强度
inline float aim_strength2 = 4.0f;

// 是否显示图像
inline std::atomic<bool> show_image{ true };

// 模型路径
//inline std::string model_path = "C:/Users/pc/Desktop/yolov10-102.onnx";
//inline std::string model_path = "C:/Users/pc/Desktop/yolov10n2.onnx";
inline std::string model_path = "C:/Users/pc/Desktop/yolov5.onnx";
//inline std::string model_path = "C:/Users/pc/Desktop/16.onnx";
//inline std::string model_path = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx";
//inline std::string model_path = "C:/Users/pc/PycharmProjects/pythonProject/models/PUBG.onnx";

#endif // GLOBAL_CONFIG_H
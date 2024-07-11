#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#include <string>
#include <atomic>
#include <nlohmann/json.hpp> // 需要安装 nlohmann/json 库
#include <fstream>

// 配置参数
inline std::string udp_ip = "192.168.1.7";
inline int udp_port = 21115;
inline float aim_strength = 1.0f;
inline float aim_strength2 = 1.0f;
inline int bullet_count = 10;
inline int shoot_range = 10;
// 是否显示图像
inline std::atomic<bool> show_image{ true };
inline std::string model_path = "./123.onnx";

// 加载配置的函数
inline void loadConfig(const std::string& configFilePath) {
    std::ifstream configFile(configFilePath);
    if (!configFile.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + configFilePath);
    }

    nlohmann::json configJson;
    configFile >> configJson;

    // 读取配置
    if (configJson.contains("udp_ip")) {
        udp_ip = configJson["udp_ip"].get<std::string>();
    }
    if (configJson.contains("udp_port")) {
        udp_port = configJson["udp_port"].get<int>();
    }
    if (configJson.contains("aim_strength")) {
        aim_strength = configJson["aim_strength"].get<float>();
    }
    if (configJson.contains("aim_strength2")) {
        aim_strength2 = configJson["aim_strength2"].get<float>();
    }
    if (configJson.contains("bullet_count")) {
        bullet_count = configJson["bullet_count"].get<int>();
    }
    if (configJson.contains("shoot_range")) {
        shoot_range = configJson["shoot_range"].get<int>();
    }
    if (configJson.contains("model_path")) {
        model_path = configJson["model_path"].get<std::string>();
    }
}

#endif // GLOBAL_CONFIG_H
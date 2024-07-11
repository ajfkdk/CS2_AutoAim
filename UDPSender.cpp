#include "UDPSender.h"
#include <cstring>
#include <chrono>
#include <iostream>

// 构造函数
UDPSender::UDPSender()
    : socket(io_context, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0)) {
}

// 析构函数
UDPSender::~UDPSender() {
    stop();
}

// 设置IP和端口
void UDPSender::setInfo(const std::string& ip, unsigned short port) {
    endpoint = boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(ip), port);
    std::cout << "Endpoint set to " << ip << ":" << port << std::endl; // 检查端点信息是否正确设置
}

// 启动发送线程
void UDPSender::start() {
    running.store(true);
    sender_thread = std::thread(&UDPSender::sendMousePosition, this);
}

// 停止发送线程
void UDPSender::stop() {
    running.store(false);
    if (sender_thread.joinable()) {
        sender_thread.join();
    }
}

// 更新鼠标位置
void UDPSender::updatePosition(int x, int y) {
    move_x.store(x);
    move_y.store(y);
    new_data_available.store(true);
}

// 发送鼠标位置
void UDPSender::sendMousePosition() {
    while (running.load()) {
        if (new_data_available.load()) {
            int x = move_x.load();
            int y = move_y.load();

            if (x != 0 || y != 0) {
                char message[9];
                message[0] = 0x01; // 消息类型

                // 将 x 和 y 值编码到消息中
                message[1] = (x >> 24) & 0xFF;
                message[2] = (x >> 16) & 0xFF;
                message[3] = (x >> 8) & 0xFF;
                message[4] = x & 0xFF;

                message[5] = (y >> 24) & 0xFF;
                message[6] = (y >> 16) & 0xFF;
                message[7] = (y >> 8) & 0xFF;
                message[8] = y & 0xFF;

                boost::system::error_code ignored_error;
                socket.send_to(boost::asio::buffer(message, sizeof(message)), endpoint, 0, ignored_error);

                new_data_available.store(false);
            }
        }
        // 为避免空转浪费 CPU 资源，添加短暂休眠
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
}

// 发送左键点击
void UDPSender::sendLeftClick() {
    char message[1];
    message[0] = 0x03; // 发送左键点击消息

    boost::system::error_code ignored_error;
    socket.send_to(boost::asio::buffer(message, sizeof(message)), endpoint, 0, ignored_error);
}
#include "UDPSender.h"
#include <cstring>
#include <chrono>

UDPSender::UDPSender(const std::string& ip, unsigned short port)
    : socket(io_context, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0)),
    endpoint(boost::asio::ip::address::from_string(ip), port) {
}

UDPSender::~UDPSender() {
    stop();
}

void UDPSender::start() {
    running.store(true);
    sender_thread = std::thread(&UDPSender::sendMousePosition, this);
}

void UDPSender::stop() {
    running.store(false);
    if (sender_thread.joinable()) {
        sender_thread.join();
    }
}

void UDPSender::updatePosition(int x, int y) {
    move_x.store(x);
    move_y.store(y);
    new_data_available.store(true);
}

void UDPSender::sendMousePosition() {
    while (running.load()) {
        if (new_data_available.load()) {
            int x = move_x.load();
            int y = move_y.load();

            if (x != 0 || y != 0) {
                char message[9];
                message[0] = 0x01; // 消息类型

                // 将x和y转换为大端序
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
    }
}
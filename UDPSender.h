#ifndef UDPSENDER_H
#define UDPSENDER_H

#include <boost/asio.hpp>
#include <atomic>
#include <thread>

class UDPSender {
public:
    UDPSender();
    ~UDPSender();

    void setInfo(const std::string& ip, unsigned short port);
    void start();
    void stop();
    void updatePosition(int x, int y);
    void sendLeftClick();

private:
    void sendMousePosition();

    boost::asio::io_context io_context;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint endpoint;

    std::atomic<bool> running{ false };
    std::atomic<int> move_x{ 0 };
    std::atomic<int> move_y{ 0 };
    std::atomic<bool> new_data_available{ false };

    std::thread sender_thread;
};

#endif // UDPSENDER_H
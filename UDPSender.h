#ifndef UDP_SENDER_H
#define UDP_SENDER_H

#include <boost/asio.hpp>
#include <thread>
#include <atomic>
#include <string>

class UDPSender {
public:
    UDPSender(const std::string& ip, unsigned short port);
    ~UDPSender();

    void start();
    void stop();
    void updatePosition(int x, int y);

private:
    void sendMousePosition();

    std::atomic<int> move_x{ 0 };
    std::atomic<int> move_y{ 0 };
    std::atomic<bool> new_data_available{ false };
    std::atomic<bool> running{ false };

    boost::asio::io_context io_context;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint endpoint;

    std::thread sender_thread;
};

#endif // UDP_SENDER_H
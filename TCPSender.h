// ---- TCPSender.h ----

#ifndef TCPSENDER_H
#define TCPSENDER_H

#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>

class TCPSender {
public:
    TCPSender(const std::string& ip, unsigned short port);
    ~TCPSender();
    void setImageBuffers(cv::Mat* writeBuffer, cv::Mat* readBuffer, std::atomic<bool>* bufferReady);
    void start();
    void stop();
    void handleMouseBox2Press();
    void handleMouseXbox1Press();
    void handleMouseBox2Release();
    void handleMouseXbox1Release();
    void setStatus(std::atomic_bool* isXbutton1, std::atomic_bool* isXbutton2);

private:
    void acceptConnections();
    void handleClient(boost::asio::ip::tcp::socket socket);

    std::string ip;
    unsigned short port;
    std::atomic<bool> running{ false };
    std::thread acceptor_thread;
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor acceptor;
    cv::Mat* writeImageBuffer;
    cv::Mat* readImageBuffer;
    std::atomic<bool>* imageBufferReady;
    std::atomic<bool> isXButton1Pressed{ false };
    std::atomic<bool> isXButton2Pressed{ false };
};

#endif // TCPSENDER_H
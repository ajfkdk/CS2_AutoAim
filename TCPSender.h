// ---- TCPSender.h ----

#ifndef TCPSENDER_H
#define TCPSENDER_H

#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <vector>

class TCPSender {
public:
    TCPSender(const std::string& imageIp, unsigned short imagePort, const std::string& keyIp, unsigned short keyPort);
    ~TCPSender();
    void start();
    void stop();
    void setImageBuffers(cv::Mat* writeBuffer, cv::Mat* readBuffer, std::atomic<bool>* bufferReady);
    void setStatus(std::atomic_bool* isXbutton1, std::atomic_bool* isXbutton2);

private:
    void acceptConnections(boost::asio::ip::tcp::acceptor& acceptor, std::function<void(boost::asio::ip::tcp::socket)> handler);
    void handleClientImage(boost::asio::ip::tcp::socket socket);
    void handleClientKey(boost::asio::ip::tcp::socket socket);
    void handleMouseXbox1Press();
    void handleMouseBox2Press();
    void handleMouseXbox1Release();
    void handleMouseBox2Release();

    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor image_acceptor;
    boost::asio::ip::tcp::acceptor key_acceptor;
    std::thread image_acceptor_thread;
    std::thread key_acceptor_thread;
    std::atomic<bool> running{ false };
    cv::Mat* writeImageBuffer{ nullptr };
    cv::Mat* readImageBuffer{ nullptr };
    std::atomic<bool>* imageBufferReady{ nullptr };
    std::atomic<bool>* isXButton1Pressed{ nullptr };
    std::atomic<bool>* isXButton2Pressed{ nullptr };
};

#endif // TCPSENDER_H
// ---- TCPSender.cpp ----

#include "TCPSender.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// 构造函数初始化 acceptor
TCPSender::TCPSender(const std::string& ip, unsigned short port)
    : acceptor(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address(ip), port)) {}

TCPSender::~TCPSender() {
    stop();
}

void TCPSender::start() {
    running = true;
    acceptor_thread = std::thread([this] { acceptConnections(); });
}

void TCPSender::stop() {
    running = false;
    io_context.stop();
    if (acceptor_thread.joinable()) {
        acceptor_thread.join();
    }
}

void TCPSender::acceptConnections() {
    while (running) {
        boost::asio::ip::tcp::socket socket(io_context);
        acceptor.accept(socket);
        std::thread(&TCPSender::handleClient, this, std::move(socket)).detach();
    }
}

void TCPSender::handleClient(boost::asio::ip::tcp::socket socket) {
    try {
        while (running) {
            auto start = std::chrono::high_resolution_clock::now();
            // 读取图像大小
            uint32_t len;
            boost::asio::read(socket, boost::asio::buffer(&len, sizeof(len)));
            len = ntohl(len);

            // 读取图像数据
            std::vector<char> buffer(len);
            boost::asio::read(socket, boost::asio::buffer(buffer.data(), buffer.size()));

            // 解码图像数据
            std::vector<uchar> data(buffer.begin(), buffer.end());
            cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);

            if (!img.empty()) {
                *writeImageBuffer = img;
                imageBufferReady->store(true, std::memory_order_release);
                std::swap(writeImageBuffer, readImageBuffer);
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "TCP IMG time: " << elapsed.count() * 1000 << " ms" << std::endl;
           
        }
    }
    catch (std::exception& e) {
        std::cerr << "处理客户端连接时发生错误: " << e.what() << std::endl;
    }
}

void TCPSender::setImageBuffers(cv::Mat* writeBuffer, cv::Mat* readBuffer, std::atomic<bool>* bufferReady) {
    writeImageBuffer = writeBuffer;
    readImageBuffer = readBuffer;
    imageBufferReady = bufferReady;
}


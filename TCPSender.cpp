// ---- TCPSender.cpp ----

#include "TCPSender.h"
#include <iostream>
#include <opencv2/opencv.hpp>

enum ProtocolType {
    MOUSE_XBOX1 = 1,
    MOUSE_BOX2 = 2,
    IMAGE = 3,
    MOUSE_XBOX1_RELEASE = 4,
    MOUSE_BOX2_RELEASE = 5
};

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

void TCPSender::handleMouseXbox1Press() {
    isXButton1Pressed.store(true, std::memory_order_release);
    std::thread([this] {
        while (isXButton1Pressed.load(std::memory_order_acquire)) {
            if (imageBufferReady->load(std::memory_order_acquire)) {
                imageBufferReady->store(false, std::memory_order_release);
                if (readImageBuffer && !readImageBuffer->empty()) {
                    // 处理图像数据的逻辑
                }
            }
        }
        }).detach();
}

void TCPSender::handleMouseBox2Press() {
    isXButton2Pressed.store(true, std::memory_order_release);
    std::thread([this] {
        while (isXButton2Pressed.load(std::memory_order_acquire)) {
            if (imageBufferReady->load(std::memory_order_acquire)) {
                imageBufferReady->store(false, std::memory_order_release);
                if (readImageBuffer && !readImageBuffer->empty()) {
                    // 处理图像数据的逻辑
                }
            }
        }
        }).detach();
}

void TCPSender::handleMouseXbox1Release() {
    isXButton1Pressed.store(false, std::memory_order_release);
}

void TCPSender::handleMouseBox2Release() {
    isXButton2Pressed.store(false, std::memory_order_release);
}

void TCPSender::handleClient(boost::asio::ip::tcp::socket socket) {
    try {
        while (running) {
            // 接收协议头
            uint32_t protocol;
            boost::asio::read(socket, boost::asio::buffer(&protocol, sizeof(protocol)));
            protocol = ntohl(protocol);

            if (protocol == ProtocolType::IMAGE) {
                // 接收图像长度
                uint32_t len;
                boost::asio::read(socket, boost::asio::buffer(&len, sizeof(len)));
                len = ntohl(len);

                // 接收图像数据
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
            }
            else if (protocol == ProtocolType::MOUSE_XBOX1) {
                std::cout << "Received mouse xbox1 press event" << std::endl;
                handleMouseXbox1Press();
            }
            else if (protocol == ProtocolType::MOUSE_BOX2) {
                std::cout << "Received mouse box2 press event" << std::endl;
                handleMouseBox2Press();
            }
            else if (protocol == ProtocolType::MOUSE_XBOX1_RELEASE) {
                std::cout << "Received mouse xbox1 release event" << std::endl;
                handleMouseXbox1Release();
            }
            else if (protocol == ProtocolType::MOUSE_BOX2_RELEASE) {
                std::cout << "Received mouse box2 release event" << std::endl;
                handleMouseBox2Release();
            }
            else {
                std::cerr << "Unknown protocol type: " << protocol << std::endl;
            }
        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception in handleClient: " << e.what() << std::endl;
    }
}

void TCPSender::setImageBuffers(cv::Mat* writeBuffer, cv::Mat* readBuffer, std::atomic<bool>* bufferReady) {
    writeImageBuffer = writeBuffer;
    readImageBuffer = readBuffer;
    imageBufferReady = bufferReady;
}

void TCPSender::setStatus(std::atomic_bool* isXbutton1, std::atomic_bool* isXbutton2) {
    isXButton1Pressed= isXbutton1;
    isXButton2Pressed = isXbutton2;
}

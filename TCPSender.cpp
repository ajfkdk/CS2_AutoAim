// ---- TCPSender.cpp ----

#include "TCPSender.h"
#include <iostream>

enum ProtocolType {
    MOUSE_XBOX1 = 1,
    MOUSE_BOX2 = 2,
    IMAGE = 3,
    MOUSE_XBOX1_RELEASE = 4,
    MOUSE_BOX2_RELEASE = 5
};

TCPSender::TCPSender(const std::string& imageIp, unsigned short imagePort, const std::string& keyIp, unsigned short keyPort)
    : image_acceptor(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address(imageIp), imagePort)),
    key_acceptor(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address(keyIp), keyPort)) {}

TCPSender::~TCPSender() {
    stop();
}

void TCPSender::start() {
    running = true;
    image_acceptor_thread = std::thread([this] { acceptConnections(image_acceptor, [this](boost::asio::ip::tcp::socket socket) { handleClientImage(std::move(socket)); }); });
    key_acceptor_thread = std::thread([this] { acceptConnections(key_acceptor, [this](boost::asio::ip::tcp::socket socket) { handleClientKey(std::move(socket)); }); });
}

void TCPSender::stop() {
    running = false;
    io_context.stop();
    if (image_acceptor_thread.joinable()) {
        image_acceptor_thread.join();
    }
    if (key_acceptor_thread.joinable()) {
        key_acceptor_thread.join();
    }
}

void TCPSender::acceptConnections(boost::asio::ip::tcp::acceptor& acceptor, std::function<void(boost::asio::ip::tcp::socket)> handler) {
    while (running) {
        boost::asio::ip::tcp::socket socket(io_context);
        acceptor.accept(socket);
        std::thread(handler, std::move(socket)).detach();
    }
}

void TCPSender::handleClientImage(boost::asio::ip::tcp::socket socket) {
    try {
        while (running) {
            // 读取图像长度
            uint32_t len;
            boost::asio::read(socket, boost::asio::buffer(&len, sizeof(len)));
            len = ntohl(len);

            // 读取图像数据
            std::vector<char> buffer(len);
            boost::asio::read(socket, boost::asio::buffer(buffer.data(), buffer.size()));

            // 处理图像数据
            std::vector<uchar> data(buffer.begin(), buffer.end());
            cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);

            if (!img.empty()) {
                *writeImageBuffer = img;
                imageBufferReady->store(true, std::memory_order_release);
                std::swap(writeImageBuffer, readImageBuffer);
            }
        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception in handleClientImage: " << e.what() << std::endl;
    }
}

void TCPSender::handleClientKey(boost::asio::ip::tcp::socket socket) {
    try {
        while (running) {
            // 读取协议类型
            uint32_t protocol;
            boost::asio::read(socket, boost::asio::buffer(&protocol, sizeof(protocol)));
            protocol = ntohl(protocol);

            if (protocol == ProtocolType::MOUSE_XBOX1) {
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
        std::cerr << "Exception in handleClientKey: " << e.what() << std::endl;
    }
}

void TCPSender::handleMouseXbox1Press() {
    isXButton1Pressed->store(true, std::memory_order_release);
}

void TCPSender::handleMouseBox2Press() {
    isXButton2Pressed->store(true, std::memory_order_release);  
}

void TCPSender::handleMouseXbox1Release() {
    isXButton1Pressed->store(false, std::memory_order_release);
}

void TCPSender::handleMouseBox2Release() {
    isXButton2Pressed->store(false, std::memory_order_release);
}

void TCPSender::setImageBuffers(cv::Mat* writeBuffer, cv::Mat* readBuffer, std::atomic<bool>* bufferReady) {
    writeImageBuffer = writeBuffer;
    readImageBuffer = readBuffer;
    imageBufferReady = bufferReady;
}

void TCPSender::setStatus(std::atomic_bool* isXbutton1, std::atomic_bool* isXbutton2) {
    isXButton1Pressed = isXbutton1;
    isXButton2Pressed = isXbutton2;
}
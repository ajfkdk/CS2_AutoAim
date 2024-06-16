#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>



void Detector(YOLO_V8*& p, const cv::Mat& input_img, cv::Mat& output_img) {
    // 将输入图像拷贝到输出图像
    output_img = input_img.clone();

    // 检测结果向量
    std::vector<DL_RESULT> res;

    // 使用YOLO模型进行检测
    p->RunSession(output_img, res);

    // 遍历检测结果
    for (auto& re : res) {
        // 随机生成一种颜色用于绘制矩形框
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // 绘制检测框
        cv::rectangle(output_img, re.box, color, 3);

        // 计算置信度（保留两位小数）
        float confidence = floor(100 * re.confidence) / 100;

        // 设置输出格式保留两位小数
        std::cout << std::fixed << std::setprecision(2);

        // 生成标签字符串，包含类别名称和置信度
        std::string label = p->classes[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        // 在检测框的上方绘制一个填充矩形，用于显示标签
        cv::rectangle(
            output_img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + label.length() * 15, re.box.y),
            color,
            cv::FILLED
        );

        // 在填充矩形内绘制标签文字
        cv::putText(
            output_img,
            label,
            cv::Point(re.box.x, re.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
    }
}

void DetectTest()
{
#define USE_CUDA
    // 创建 YOLO_V8 检测器对象
    YOLO_V8* yoloDetector = new YOLO_V8;

    
    yoloDetector->classes = { "person","head" };

    // 初始化检测参数
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.5; // 设置矩形框置信度阈值
    params.iouThreshold = 0.5; // 设置 IOU 阈值
    //params.modelPath = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx"; // 设置模型路径
    params.modelPath = "C:/Users/pc/PycharmProjects/pythonProject/models/PUBG.onnx"; // 设置模型路径
    //params.modelPath = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx"; // 设置模型路径
    params.imgSize = { INPUT_SIZE, INPUT_SIZE }; // 设置输入图像的大小

#ifdef USE_CUDA
    // 启用 CUDA
    params.cudaEnable = true;

    // 使用 GPU 进行 FP32 推理
    params.modelType = YOLO_DETECT_V8;
    // 使用 GPU 进行 FP16 推理（需要更换 FP16 模型）
    // Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // 使用 CPU 进行推理
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif
    auto img_path = "C:/Users/pc/PycharmProjects/pythonProject/img.png";
    cv::Mat img = cv::imread(img_path);
    cv::Mat output_img;
    // 创建 YOLO 会话
    yoloDetector->CreateSession(params);

    // 调用检测函数进行图像检测
    Detector(yoloDetector, img, output_img);
    cv::imshow("output", output_img);
    cv::waitKey(0);
}


int main()
{
    DetectTest();
    // ClsTest();
}
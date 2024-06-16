#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

#include <regex>


void Detector(YOLO_V8*& p) {
    // 获取当前工作路径
    std::filesystem::path current_path = std::filesystem::current_path();
    // 拼接出图像文件夹路径
    std::filesystem::path imgs_path = current_path / "images";

    // 如果图像文件夹不存在
    if (!std::filesystem::exists(imgs_path))
	{
		// 提示用户图像文件夹不存在
		std::cerr << "Images folder does not exist" << std::endl;
		return;
	}

    // 遍历图像文件夹中的所有文件
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        // 如果文件的扩展名是.jpg, .png 或 .jpeg
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            // 获取图像路径并加载图像
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;

            // 使用YOLO模型进行检测
            p->RunSession(img, res);

            // 遍历检测结果
            for (auto& re : res)
            {
                // 随机生成一种颜色用于绘制矩形框
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                // 绘制检测框
                cv::rectangle(img, re.box, color, 3);

                // 计算置信度（保留两位小数）
                float confidence = floor(100 * re.confidence) / 100;

                // 设置输出格式保留两位小数
                std::cout << std::fixed << std::setprecision(2);

                // 生成标签字符串，包含类别名称和置信度
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                // 在检测框的上方绘制一个填充矩形，用于显示标签
                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                // 在填充矩形内绘制标签文字
                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );
            }

            // 提示用户按任意键退出
            std::cout << "Press any key to exit" << std::endl;

            // 显示检测结果并等待用户按键
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);

            // 关闭窗口
            cv::destroyAllWindows();
        }
    }
}


int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("C:/Users/pc/Downloads/coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}



void DetectTest()
{
    // 创建 YOLO_V8 检测器对象
    YOLO_V8* yoloDetector = new YOLO_V8;

    // 读取 COCO 数据集的配置文件
    ReadCocoYaml(yoloDetector);

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

    // 创建 YOLO 会话
    yoloDetector->CreateSession(params);

    // 调用检测函数进行图像检测
    Detector(yoloDetector);
}


int main()
{
    DetectTest();
    // ClsTest();
}
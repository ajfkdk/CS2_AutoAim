#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

#include <regex>


void Detector(YOLO_V8*& p) {
    // ��ȡ��ǰ����·��
    std::filesystem::path current_path = std::filesystem::current_path();
    // ƴ�ӳ�ͼ���ļ���·��
    std::filesystem::path imgs_path = current_path / "images";

    // ���ͼ���ļ��в�����
    if (!std::filesystem::exists(imgs_path))
	{
		// ��ʾ�û�ͼ���ļ��в�����
		std::cerr << "Images folder does not exist" << std::endl;
		return;
	}

    // ����ͼ���ļ����е������ļ�
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        // ����ļ�����չ����.jpg, .png �� .jpeg
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            // ��ȡͼ��·��������ͼ��
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;

            // ʹ��YOLOģ�ͽ��м��
            p->RunSession(img, res);

            // ���������
            for (auto& re : res)
            {
                // �������һ����ɫ���ڻ��ƾ��ο�
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                // ���Ƽ���
                cv::rectangle(img, re.box, color, 3);

                // �������Ŷȣ�������λС����
                float confidence = floor(100 * re.confidence) / 100;

                // ���������ʽ������λС��
                std::cout << std::fixed << std::setprecision(2);

                // ���ɱ�ǩ�ַ���������������ƺ����Ŷ�
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                // �ڼ�����Ϸ�����һ�������Σ�������ʾ��ǩ
                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                // ���������ڻ��Ʊ�ǩ����
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

            // ��ʾ�û���������˳�
            std::cout << "Press any key to exit" << std::endl;

            // ��ʾ��������ȴ��û�����
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);

            // �رմ���
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
    // ���� YOLO_V8 ���������
    YOLO_V8* yoloDetector = new YOLO_V8;

    // ��ȡ COCO ���ݼ��������ļ�
    ReadCocoYaml(yoloDetector);

    // ��ʼ��������
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.5; // ���þ��ο����Ŷ���ֵ
    params.iouThreshold = 0.5; // ���� IOU ��ֵ
    //params.modelPath = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx"; // ����ģ��·��
    params.modelPath = "C:/Users/pc/PycharmProjects/pythonProject/models/PUBG.onnx"; // ����ģ��·��
    //params.modelPath = "C:/Users/pc/PycharmProjects/pythonProject/yolov8n.onnx"; // ����ģ��·��
    params.imgSize = { INPUT_SIZE, INPUT_SIZE }; // ��������ͼ��Ĵ�С

#ifdef USE_CUDA
    // ���� CUDA
    params.cudaEnable = true;

    // ʹ�� GPU ���� FP32 ����
    params.modelType = YOLO_DETECT_V8;
    // ʹ�� GPU ���� FP16 ������Ҫ���� FP16 ģ�ͣ�
    // Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // ʹ�� CPU ��������
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif

    // ���� YOLO �Ự
    yoloDetector->CreateSession(params);

    // ���ü�⺯������ͼ����
    Detector(yoloDetector);
}


int main()
{
    DetectTest();
    // ClsTest();
}
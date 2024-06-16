#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>



void Detector(YOLO_V8*& p, const cv::Mat& input_img, cv::Mat& output_img) {
    // ������ͼ�񿽱������ͼ��
    output_img = input_img.clone();

    // ���������
    std::vector<DL_RESULT> res;

    // ʹ��YOLOģ�ͽ��м��
    p->RunSession(output_img, res);

    // ���������
    for (auto& re : res) {
        // �������һ����ɫ���ڻ��ƾ��ο�
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // ���Ƽ���
        cv::rectangle(output_img, re.box, color, 3);

        // �������Ŷȣ�������λС����
        float confidence = floor(100 * re.confidence) / 100;

        // ���������ʽ������λС��
        std::cout << std::fixed << std::setprecision(2);

        // ���ɱ�ǩ�ַ���������������ƺ����Ŷ�
        std::string label = p->classes[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        // �ڼ�����Ϸ�����һ�������Σ�������ʾ��ǩ
        cv::rectangle(
            output_img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + label.length() * 15, re.box.y),
            color,
            cv::FILLED
        );

        // ���������ڻ��Ʊ�ǩ����
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
    // ���� YOLO_V8 ���������
    YOLO_V8* yoloDetector = new YOLO_V8;

    
    yoloDetector->classes = { "person","head" };

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
    auto img_path = "C:/Users/pc/PycharmProjects/pythonProject/img.png";
    cv::Mat img = cv::imread(img_path);
    cv::Mat output_img;
    // ���� YOLO �Ự
    yoloDetector->CreateSession(params);

    // ���ü�⺯������ͼ����
    Detector(yoloDetector, img, output_img);
    cv::imshow("output", output_img);
    cv::waitKey(0);
}


int main()
{
    DetectTest();
    // ClsTest();
}
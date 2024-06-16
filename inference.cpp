#include "inference.h"
#include <regex>

#define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))
YOLO_V8::YOLO_V8() {

}


YOLO_V8::~YOLO_V8() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}

char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    // 判断输入图像的通道数是否为3（即彩色图像）
    if (iImg.channels() == 3)
    {
        // 克隆输入图像到输出图像
        oImg = iImg.clone();
        // 将图像从BGR颜色空间转换为RGB颜色空间
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        // 将灰度图像转换为RGB颜色空间
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }
    // 如果图像宽度大于或等于高度
    if (iImg.cols >= iImg.rows)
    {
        // 计算缩放比例
        resizeScales = iImg.cols / (float)iImgSize.at(0);
        // 按比例调整图像大小
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
    }
    else
    {
        // 计算缩放比例
        resizeScales = iImg.rows / (float)iImgSize.at(0);
        // 按比例调整图像大小
        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
    }
    // 创建一个指定大小的全零矩阵（黑色图像）
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
    // 将调整大小后的图像复制到全零矩阵的左上角
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    // 将输出图像设置为新的全零矩阵
    oImg = tempImg;
   

    // 返回成功状态
    return RET_OK;
}

char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    // 定义返回值并初始化为成功状态
    char* Ret = RET_OK;

    // 定义正则表达式模式，用于检测中文字符
    std::regex pattern("[\u4e00-\u9fa5]");

    // 使用正则表达式搜索模型路径中的中文字符
    bool result = std::regex_search(iParams.modelPath, pattern);

    // 如果模型路径包含中文字符，返回错误信息
    if (result)
    {
        Ret = "[YOLO_V8]:Your model path is error. Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }

    try
    {
        // 初始化类成员变量
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;

        // 创建ORT环境
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;

        // 如果启用CUDA，则设置CUDA选项
        if (iParams.cudaEnable)
        {
            cudaEnable = iParams.cudaEnable;
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }

        // 设置图优化级别
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 设置线程数和日志级别
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        // 如果在Windows系统上，处理宽字符路径
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        // 如果在非Windows系统上，直接使用字符路径
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        // 创建ORT会话
        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;

        // 获取输入节点数目并保存输入节点名称
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }

        // 获取输出节点数目并保存输出节点名称
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }

        // 设置运行选项并进行预热
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();

        return RET_OK; // 返回成功状态
    }
    catch (const std::exception& e)
    {
        // 捕获异常并返回错误信息
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[YOLO_V8]:Create session failed.";
    }
}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    // 记录开始时间用于基准测试
    clock_t starttime_1 = clock();
#endif // benchmark

    // 定义返回值并初始化为成功状态
    char* Ret = RET_OK;

    // 定义处理后的图像
    cv::Mat processedImg;

    // 调用预处理函数对输入图像进行预处理
    PreProcess(iImg, imgSize, processedImg);

    // 根据模型类型进行不同的处理
    if (modelType < 4)
    {
        // 为存储预处理图像数据分配内存
        float* blob = new float[processedImg.total() * 3];

        // 将图像数据转换为blob格式
        BlobFromImage(processedImg, blob);

        // 定义输入节点的维度
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };

        // 调用TensorProcess函数进行推理
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);

 
    }
    else
    {
#ifdef USE_CUDA
        // 为存储预处理图像数据分配内存（半精度浮点数）
        half* blob = new half[processedImg.total() * 3];

        // 将图像数据转换为blob格式
        BlobFromImage(processedImg, blob);

        // 定义输入节点的维度
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };

        // 调用TensorProcess函数进行推理
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);

        // 释放内存
        delete[] blob;
#endif
    }

    // 返回成功状态
    return Ret;
}

template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult) {
    // 创建输入张量
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());

#ifdef benchmark
    // 记录推理开始时间
    clock_t starttime_2 = clock();
#endif // benchmark

    // 运行推理会话
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());

#ifdef benchmark
    // 记录推理结束时间
    clock_t starttime_3 = clock();
#endif // benchmark

    // 获取输出张量信息
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();

    // 释放输入数据内存
    delete[] blob;

    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    {
        int strideNum = outputNodeDims[2]; // 获取输出节点维度的第三个值，表示步幅数量
        int signalResultNum = outputNodeDims[1]; // 获取输出节点维度的第二个值，表示信号结果数量
        std::vector<int> class_ids; // 用于存储类别ID的向量
        std::vector<float> confidences; // 用于存储置信度的向量
        std::vector<cv::Rect> boxes; // 用于存储检测到的矩形框的向量

        cv::Mat rawData; // 用于存储原始数据的矩阵
        if (modelType == 1) {
            // 如果模型类型为1，则使用FP32格式
            rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
        }
        else {
            // 否则使用FP16格式，并将其转换为FP32格式
            rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        rawData = rawData.t(); // 转置矩阵
        float* data = (float*)rawData.data; // 获取矩阵数据的指针

  
        for (int i = 0; i < strideNum; ++i) {

            float* classesScores = data + 4; // 获取类别得分的起始位置
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores); // 创建包含类别得分的矩阵

            cv::Point class_id; // 用于存储类别ID的点
            double maxClassScore; // 用于存储最大类别得分

            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id); // 查找最大类别得分及其对应的类别ID

            if (maxClassScore > rectConfidenceThreshold) {

                // 如果最大类别得分超过置信度阈值，则保存该结果
                confidences.push_back(maxClassScore); 
                class_ids.push_back(class_id.x); 

                // 获取边框的中心坐标和宽高
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                // 计算边框的左上角坐标
                int left = int((x - 0.5 * w) * resizeScales);
                int top = int((y - 0.5 * h) * resizeScales);

                // 计算边框的宽和高
                int width = int(w * resizeScales);
                int height = int(h * resizeScales);

                // 存储边框
                boxes.emplace_back(left, top, width, height);
            }
            data += signalResultNum; // 移动到下一个结果
        }

        // 执行非极大值抑制（NMS）
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);

        // 保存最终的检测结果
        for (int i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }
#ifdef benchmark
        // 记录后处理结束时间
        clock_t starttime_4 = clock();

        // 计算处理时间
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;

        // 输出处理时间
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
        else
        {
            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
#endif // benchmark

        break;
    }

    return RET_OK;

    }
}


char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
            outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return RET_OK;
}

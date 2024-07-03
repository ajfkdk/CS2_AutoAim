#include "inference.h"
#include <regex>

//#define benchmark
#define USE_CUDA
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

    if (!iImg.isContinuous()) {
        iImg = iImg.clone();
    }

    const uchar* imgData = iImg.data;
    float scale = 1.0f / 255.0f;

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < imgHeight; ++h) {
            for (int w = 0; w < imgWidth; ++w) {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] =
                    typename std::remove_pointer<T>::type(
                        imgData[h * imgWidth * channels + w * channels + c] * scale);
            }
        }
    }
    return RET_OK;
}

char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    // �ж�����ͼ���ͨ�����Ƿ�Ϊ3������ɫͼ��
    if (iImg.channels() == 3)
    {
        // ��¡����ͼ�����ͼ��
        oImg = iImg.clone();
        // ��ͼ���BGR��ɫ�ռ�ת��ΪRGB��ɫ�ռ�
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        // ���Ҷ�ͼ��ת��ΪRGB��ɫ�ռ�
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }
    // ���ͼ���ȴ��ڻ���ڸ߶�
    if (iImg.cols >= iImg.rows)
    {
        // �������ű���
        resizeScales = iImg.cols / (float)iImgSize.at(0);
        // ����������ͼ���С
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
    }
    else
    {
        // �������ű���
        resizeScales = iImg.rows / (float)iImgSize.at(0);
        // ����������ͼ���С
        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
    }
    // ����һ��ָ����С��ȫ����󣨺�ɫͼ��
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
    // ��������С���ͼ���Ƶ�ȫ���������Ͻ�
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    // �����ͼ������Ϊ�µ�ȫ�����
    oImg = tempImg;
   

    // ���سɹ�״̬
    return RET_OK;
}

char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    // ���巵��ֵ����ʼ��Ϊ�ɹ�״̬
    char* Ret = RET_OK;

    // ����������ʽģʽ�����ڼ�������ַ�
    std::regex pattern("[\u4e00-\u9fa5]");

    // ʹ��������ʽ����ģ��·���е������ַ�
    bool result = std::regex_search(iParams.modelPath, pattern);

    // ���ģ��·�����������ַ������ش�����Ϣ
    if (result)
    {
        Ret = "[YOLO_V8]:Your model path is error. Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }

    try
    {
        // ��ʼ�����Ա����
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;

        // ����ORT����
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
         
        // �������CUDA��������CUDAѡ��
        if (iParams.cudaEnable)
        {
            cudaEnable = iParams.cudaEnable;
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }

        // ����ͼ�Ż�����
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // �����߳�������־����
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        // �����Windowsϵͳ�ϣ�������ַ�·��
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        // ����ڷ�Windowsϵͳ�ϣ�ֱ��ʹ���ַ�·��
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        // ����ORT�Ự
        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;

        // ��ȡ����ڵ���Ŀ����������ڵ�����
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }

        // ��ȡ����ڵ���Ŀ����������ڵ�����
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }

        // ��������ѡ�����Ԥ��
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();

        return RET_OK; // ���سɹ�״̬
    }
    catch (const std::exception& e)
    {
        // �����쳣�����ش�����Ϣ
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
    // ��¼��ʼʱ�����ڻ�׼����
    clock_t starttime_1 = clock();
    //clock_t starttime_1 = clock();
    // ���巵��ֵ����ʼ��Ϊ�ɹ�״̬
    char* Ret = RET_OK;

    // ���崦����ͼ��
    cv::Mat processedImg;

    // ����Ԥ������������ͼ�����Ԥ����
    PreProcess(iImg, imgSize, processedImg);
    // Ϊ�洢Ԥ����ͼ�����ݷ����ڴ�
    float* blob = new float[processedImg.total() * 3];
   
    // ��ͼ������ת��Ϊblob��ʽ
    BlobFromImage(processedImg, blob);
    
    // ��������ڵ��ά��
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    // ����TensorProcess������������
    TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    
    // ���سɹ�״̬
    return Ret;
}

template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DL_RESULT>& oResult) {
    // ������������
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());

#ifdef benchmark
    // ��¼����ʼʱ��
    clock_t starttime_2 = clock();
#endif // benchmark

    // ��������Ự
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());

#ifdef benchmark
    // ��¼�������ʱ��
    clock_t starttime_3 = clock();
#endif // benchmark

    // ��ȡ���������Ϣ
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();

    // �ͷ����������ڴ�
    delete[] blob;

    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    {
        
        int numDetections = outputNodeDims[2]; // ��ʾ��������� 8400
        int numAttributes = outputNodeDims[1]; // ��ʾÿ�����������������8��

        std::vector<int> class_ids; // ���ڴ洢���ID������
        std::vector<float> confidences; // ���ڴ洢���Ŷȵ�����
        std::vector<cv::Rect> boxes; // ���ڴ洢��⵽�ľ��ο������

        cv::Mat rawData; // ���ڴ洢ԭʼ���ݵľ���
        if (modelType == YOLO_DETECT_V8) {
            // ���ģ������ΪYOLO_DETECT_V8����ʹ��FP32��ʽ
            rawData = cv::Mat(numAttributes, numDetections, CV_32F, output);
        }
        else {
            // ����ʹ��FP16��ʽ��������ת��ΪFP32��ʽ
            rawData = cv::Mat(numDetections, numAttributes, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        // ת�ú�ѹ������
        cv::Mat transposedData;
        cv::transpose(rawData, transposedData);
        transposedData = transposedData.reshape(1, { numDetections, numAttributes });

        float* data = (float*)transposedData.data; // ��ȡת�ú�������ݵ�ָ��

        for (int i = 0; i < numDetections; ++i) {
            // ��ȡ������Ŷ�
            float* classes_scores = data + 4;
            float max_score = *std::max_element(classes_scores, classes_scores + (numAttributes - 4));

            // ������Ŷȳ�����ֵ������ü���
            if (max_score >= rectConfidenceThreshold) {
                int class_id = std::distance(classes_scores, std::max_element(classes_scores, classes_scores + (numAttributes - 4)));

                // ��ȡ�߽������
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                // ����߿�����Ͻ�����
                int left = int((x - 0.5 * w) * resizeScales);
                int top = int((y - 0.5 * h) * resizeScales);

                // ����߿�Ŀ�͸�
                int width = int(w * resizeScales);
                int height = int(h * resizeScales);

                // �洢�߿�
                boxes.emplace_back(left, top, width, height);
                confidences.push_back(max_score);
                class_ids.push_back(class_id);
            }
            data += numAttributes; // �ƶ�����һ������
        }

        std::vector<int> nmsResult;
        // ȷ���ڵ��� NMSBoxes ֮ǰ��boxes �� confidences �ĳ���һ��
        if (boxes.size() == confidences.size()) {
            // ִ�зǼ���ֵ���ƣ�NMS��
            cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);

            // �������յļ����
            for (int i = 0; i < nmsResult.size(); ++i) {
                int idx = nmsResult[i];
                DL_RESULT result;
                result.classId = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];
                oResult.push_back(result);
            }
        }
        else {
            std::cerr << "Error: The number of boxes and confidences must be equal before applying NMS." << std::endl;
            return RET_OK;
        }

  
#ifdef benchmark
        // ��¼�������ʱ��
        clock_t starttime_4 = clock();

        // ���㴦��ʱ��
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;

        // �������ʱ��
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
    case YOLO_DETECT_V10:
    {
        int numDetections = outputNodeDims[1]; // ��ʾ��������� 300
        int numAttributes = outputNodeDims[2]; // ��ʾÿ�����������������6��


        for (int i = 0; i < numDetections; ++i) {
            // ��ȡ��ǰ���������
            float x1 = output[i * numAttributes + 0];
            float y1 = output[i * numAttributes + 1];
            float x2 = output[i * numAttributes + 2];
            float y2 = output[i * numAttributes + 3];
            float score = output[i * numAttributes + 4];
            int class_id = static_cast<int>(output[i * numAttributes + 5]);

            // ������Ŷȳ�����ֵ������ü���
            if (score >= rectConfidenceThreshold) {
                // ����߿�����Ͻ�����Ϳ��
                int left = static_cast<int>(x1 * resizeScales);
                int top = static_cast<int>(y1 * resizeScales);
                int width = static_cast<int>((x2 - x1) * resizeScales);
                int height = static_cast<int>((y2 - y1) * resizeScales);

                // �洢�����

                DL_RESULT result;
                result.classId = class_id;
                result.confidence = score;
                result.box = cv::Rect(left, top, width, height);
                oResult.push_back(result);
            }
        }
#ifdef benchmark
        // ��¼�������ʱ��
        clock_t starttime_4 = clock();

        // ���㴦��ʱ��
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;

        // �������ʱ��
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
        else
        {
            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
#endif // benchmark
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

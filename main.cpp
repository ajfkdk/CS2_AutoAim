#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <array>
#include <iostream>

class YOLOV5 {
public:
    YOLOV5(const std::string& model_path) {
        // Initialize ONNX Runtime
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOV5");
        Ort::SessionOptions session_options;

        // Enable CUDA if available
#ifdef USE_CUDA
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Convert std::string to ORTCHAR_T*
#ifdef _WIN32
        std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());
        const ORTCHAR_T* ort_model_path = w_model_path.c_str();
#else
        const ORTCHAR_T* ort_model_path = model_path.c_str();
#endif

        // Initialize ONNX Runtime session
        session = Ort::Session(env, ort_model_path, session_options);

        // Get input and output node names
        Ort::AllocatorWithDefaultOptions allocator;
        numInputNodes = session.GetInputCount();
        numOutputNodes = session.GetOutputCount();

        for (size_t i = 0; i < numInputNodes; ++i) {
            Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());

            auto input_type_info = session.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_w = input_dims[3];
            input_h = input_dims[2];
        }

        for (size_t i = 0; i < numOutputNodes; ++i) {
            Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());
        }
    }

    std::vector<float> inference(cv::Mat& img) {
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(input_w, input_h));
        resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

        std::vector<int64_t> input_shape = { 1, 3, input_h, input_w };
        size_t input_tensor_size = input_h * input_w * 3;
        std::vector<float> input_tensor_values(input_tensor_size);
        std::memcpy(input_tensor_values.data(), resized_img.data, input_tensor_size * sizeof(float));

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

        const char* const input_names[] = { input_node_names[0].c_str() };
        const char* const output_names[] = { output_node_names[0].c_str() };

        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
        float* float_array = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape_info = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

        std::vector<float> result(float_array, float_array + output_shape_info[1] * output_shape_info[2]);
        return result;
    }

private:
    Ort::Env env{ nullptr };
    Ort::Session session{ nullptr };
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes;
    size_t numOutputNodes;
    int input_w;
    int input_h;
};

int main() {
    YOLOV5 yolo("C:/Users/pc/Downloads/LANShareDownloads/result/models/PUBG.onnx");
    cv::Mat img = cv::imread("C:/Users/pc/Downloads/LANShareDownloads/result/img_2.png");
    std::vector<float> result = yolo.inference(img);
    cv::waitKey(1);
    // Process result...

    return 0;
}
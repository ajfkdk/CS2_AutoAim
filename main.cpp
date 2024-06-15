#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <array>
#include <numeric>
#include <iostream>

using namespace std;
using namespace cv;

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
        // 调整图像尺寸到模型输入要求的宽高
        cv::resize(img, resized_img, cv::Size(input_w, input_h));
        // 将图像像素值归一化到[0, 1]范围，并转换为浮点数类型
        resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
        // 将图像从BGR格式转换为RGB格式
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

        // 定义输入张量的形状，形状为{batch_size, channels, height, width}
        std::vector<int64_t> input_shape = { 1, 3, input_h, input_w };
        // 计算输入张量的大小
        size_t input_tensor_size = input_h * input_w * 3;
        // 创建存储输入张量数据的向量
        std::vector<float> input_tensor_values(input_tensor_size);
        // 将图像数据拷贝到输入张量的向量中
        std::memcpy(input_tensor_values.data(), resized_img.data, input_tensor_size * sizeof(float));

        // 创建CPU内存信息，用于分配内存
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 创建输入张量
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

        // 定义输入和输出节点的名称
        const char* const input_names[] = { input_node_names[0].c_str() };
        const char* const output_names[] = { output_node_names[0].c_str() };

        // 运行模型推理，获取输出张量
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
        // 获取输出张量的数据指针
        float* float_array = output_tensors.front().GetTensorMutableData<float>();
        // 获取输出张量的形状信息
        auto output_shape_info = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

        // 将输出数据复制到结果向量中并返回
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

vector<int> nms(const vector<vector<float>>& dets, float thresh) {
    vector<int> keep;
    vector<float> x1, y1, x2, y2, scores;
    for (const auto& det : dets) {
        x1.push_back(det[0]);
        y1.push_back(det[1]);
        x2.push_back(det[2]);
        y2.push_back(det[3]);
        scores.push_back(det[4]);
    }

    vector<int> order(scores.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        vector<int> inds;
        for (size_t j = 1; j < order.size(); ++j) {
            int k = order[j];
            float xx1 = max(x1[i], x1[k]);
            float yy1 = max(y1[i], y1[k]);
            float xx2 = min(x2[i], x2[k]);
            float yy2 = min(y2[i], y2[k]);
            float w = max(0.0f, xx2 - xx1);
            float h = max(0.0f, yy2 - yy1);
            float inter = w * h;
            float iou = inter / ((x2[i] - x1[i]) * (y2[i] - y1[i]) + (x2[k] - x1[k]) * (y2[k] - y1[k]) - inter);
            if (iou <= thresh) {
                inds.push_back(k);
            }
        }
        order = inds;
    }
    return keep;
}

vector<vector<float>> xywh2xyxy(const vector<vector<float>>& x) {
    vector<vector<float>> y(x.size(), vector<float>(4));
    for (size_t i = 0; i < x.size(); ++i) {
        y[i][0] = x[i][0] - x[i][2] / 2;
        y[i][1] = x[i][1] - x[i][3] / 2;
        y[i][2] = x[i][0] + x[i][2] / 2;
        y[i][3] = x[i][1] + x[i][3] / 2;
    }
    return y;
}

pair<int, int> find_most_nearby_bbox(const vector<vector<float>>& boxes, int capture_size, float target_adjustment, int image_top_left_x, int image_top_left_y) {
    int target_x = capture_size / 2;
    int target_y = capture_size / 2;
    float min_distance = numeric_limits<float>::infinity();
    int nearest_bbox_x = target_x;
    int nearest_bbox_y = target_y;

    for (const auto& bbox : boxes) {
        int bbox_x = (bbox[0] + bbox[2]) / 2;
        int bbox_y = (bbox[1] + bbox[3]) / 2;
        int adjusted_bbox_y = bbox_y - static_cast<int>((bbox[3] - bbox[1]) * target_adjustment);
        float distance = pow(bbox_x - target_x, 2) + pow(adjusted_bbox_y - target_y, 2);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_bbox_x = bbox_x;
            nearest_bbox_y = adjusted_bbox_y;
        }
    }
    return { nearest_bbox_x + image_top_left_x, nearest_bbox_y + image_top_left_y };
}


pair<int, int> find_most_nearby_bbox(const vector<vector<float>>& boxes, float target_adjustment, int capture_size, int image_top_left_x, int image_top_left_y) {
    int target_x = capture_size / 2;
    int target_y = capture_size / 2;
    float min_distance = numeric_limits<float>::infinity();
    int nearest_bbox_x = target_x;
    int nearest_bbox_y = target_y;

    for (const auto& bbox : boxes) {
        int bbox_x = (bbox[0] + bbox[2]) / 2;
        int bbox_y = (bbox[1] + bbox[3]) / 2;

        int adjusted_bbox_y = bbox_y - static_cast<int>((bbox[3] - bbox[1]) * target_adjustment);

        float distance = pow(bbox_x - target_x, 2) + pow(adjusted_bbox_y - target_y, 2);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_bbox_x = bbox_x;
            nearest_bbox_y = adjusted_bbox_y;
        }
    }

    return { nearest_bbox_x + image_top_left_x, nearest_bbox_y + image_top_left_y };
}

vector<vector<float>> xywh2xyxy(const vector<vector<float>>& x);


vector<vector<float>> filter_box(const vector<vector<float>>& org_box, float conf_thres, float iou_thres) {
    vector<vector<float>> output;  // 用于存储最终输出的过滤后框
    vector<vector<float>> box;  // 用于存储置信度高于阈值的框
    float top_conf = 0.0;  // 用于存储最高的置信度
    // 遍历原始框，过滤掉置信度低于阈值的框
    for (const auto& b : org_box) {
        if(tan(b[4]) > top_conf) {
			top_conf = tan(b[4]);
		}
        if (b[4] > conf_thres) {

            box.push_back(b);
        }
    }

    vector<int> cls(box.size());  // 用于存储每个框的类别
    // 计算每个框的类别，并更新框的类别信息
    for (size_t i = 0; i < box.size(); ++i) {
        cls[i] = max_element(box[i].begin() + 5, box[i].end()) - box[i].begin() - 5;
        box[i][5] = cls[i];
    }

    set<int> all_cls(cls.begin(), cls.end());  // 获取所有类别的集合
    // 对每个类别分别进行处理
    for (int curr_cls : all_cls) {
        vector<vector<float>> curr_cls_box;  // 当前类别的框
        // 筛选出当前类别的所有框
        for (const auto& b : box) {
            if (b[5] == curr_cls) {
                curr_cls_box.push_back(b);
            }
        }

        // 将当前类别的框从xywh格式转换为xyxy格式
        curr_cls_box = xywh2xyxy(curr_cls_box);
        // 对当前类别的框进行非极大值抑制，获取保留的框的索引
        vector<int> curr_out_box_indices = nms(curr_cls_box, iou_thres);
        printf("curr_out_box_indices.size() = %d\n top_conf = %f\n", curr_out_box_indices.size(), top_conf);
        // 根据索引将保留的框加入输出
        for (int idx : curr_out_box_indices) {
            output.push_back(curr_cls_box[idx]);
        }
    }

    return output;  // 返回过滤后的框
}

void process_image(Mat& image, YOLOV5& model, bool showImage, int screen_width, int screen_height, int capture_size, Scalar bbox_color, int bbox_thickness, const vector<string>& classes, float target_adjustment, int image_top_left_x, int image_top_left_y) {
    vector<float> result_flat = model.inference(image);

    // Assuming each box is represented by 6 values: [x1, y1, x2, y2, confidence, class]
    const int box_size = 6;
    vector<vector<float>> result;
    for (size_t i = 0; i < result_flat.size(); i += box_size) {
        vector<float> box(result_flat.begin() + i, result_flat.begin() + i + box_size);
        result.push_back(box);
    }

    vector<vector<float>> outbox = filter_box(result, 0.7, 0.5);

    if (outbox.empty()) {
        return;
    }

    if (showImage) {
        for (const auto& box : outbox) {
            int top = static_cast<int>(box[1]); // y1
            int left = static_cast<int>(box[0]); // x1
            int right = static_cast<int>(box[2]); // x2
            int bottom = static_cast<int>(box[3]); // y2
            float score = box[4];
            int cl = static_cast<int>(box[5]);

            rectangle(image, Point(left, top), Point(right, bottom), bbox_color, bbox_thickness);
            putText(image, format("%s %.2f", classes[cl].c_str(), score), Point(left, top), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(33, 66, 131), bbox_thickness);
        }
    }

    pair<int, int> nearest_box_xy = find_most_nearby_bbox(outbox, target_adjustment, capture_size, image_top_left_x, image_top_left_y);
    cout << "Nearest box coordinates: (" << nearest_box_xy.first << ", " << nearest_box_xy.second << ")" << endl;
}
int main() {
    // Assuming you have the necessary model path and other variables defined
    string onnx_path = "C:/Users/pc/Downloads/LANShareDownloads/result/models/PUBG.onnx";
    YOLOV5 model(onnx_path);

    string image_path = "C:/Users/pc/Desktop/Snipaste_2024-06-15_22-37-13.png";
    Mat captured_image = imread(image_path);

    bool showImage = true;
    int screen_width = 1920;
    int screen_height = 1080;
    int capture_size = 320;
    Scalar bbox_color(203, 219, 120);  // BGR color
    int bbox_thickness = 2;
    vector<string> classes = { "player", "head" };
    float target_adjustment = 0.5;
    int image_top_left_x = (screen_width - capture_size) / 2;
    int image_top_left_y = (screen_height - capture_size) / 2;

    process_image(captured_image, model, showImage, screen_width, screen_height, capture_size, bbox_color, bbox_thickness, classes, target_adjustment, image_top_left_x, image_top_left_y);

    if (showImage) {
        imshow("processed_image", captured_image);
        waitKey(50000);
    }

    cout << "AI Process Exit" << endl;
    return 0;
}
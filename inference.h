#pragma once

#define    RET_OK nullptr
#define    USE_CUDA
#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "global_config.h"
#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif



enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6,

    YOLO_DETECT_V10 = 7,
};


typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V10;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.8;
    float iouThreshold = 0.5;
    int	keyPointsNum = 2;//Note:kpt number for pose
    bool cudaEnable = true;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 4;
} DL_INIT_PARAM;





class YOLO_V8
{
public:
    YOLO_V8();

    ~YOLO_V8();

public:
    char* CreateSession(DL_INIT_PARAM& iParams);

    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

    char* WarmUpSession();

    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes{};

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    // 添加这些成员变量
    float* blob;
    size_t blobSize;
    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale
};

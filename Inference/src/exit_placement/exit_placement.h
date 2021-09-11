#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "parserOnnxConfig.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <torch/torch.h>


class Profiler {
    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    using Severity = nvinfer1::ILogger::Severity;

public:
    Profiler(
      const size_t& batch_size_s1 = 1,
      const size_t& batch_size_s2 = 1,
      const int batch_num = 1,
      const Severity severity = Severity::kWARNING);

    bool build_s1();
    bool build_s2();
    int batch_num_;
    bool infer(const size_t& num_test, const int batch_idx, const int copy_method);

private:
    nvinfer1::DataType model_dtype_;
    bool fp16_{false};
    bool int8_{false};

    size_t batch_size_s1_;
    size_t batch_size_s2_;
    cudaStream_t stream_;
    cudaEvent_t infer_start;
    cudaEvent_t s1_end;
    cudaEvent_t s2_end;
    nvinfer1::Dims input_dims_s1;
    nvinfer1::Dims input_dims_s2;
    nvinfer1::Dims output_dims_s1;
    nvinfer1::Dims output_dims_s2;
    std::string input_tensor_names_;
    std::string output_tensor_names_;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_s1;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_s2;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext_s1;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext_s2;
    // samplesCommon::BufferManager mBufferManager;
    bool construct_s1(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser);
    bool construct_s2(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser);
};
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
#include <torch/torch.h>

class ProfilerConfig {
public:
    ProfilerConfig() {}
    ~ProfilerConfig() {}
}

class Profiler {
    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    using Severity = nvinfer1::ILogger::Severity;

public:
    Profiler(
      const ProfilerConfig& profiler_config, const size_t& min_batch_size = 1,
      const size_t& opt_batch_size = 1, const size_t& max_batch_size = 1, const int batch_num = 1,
      const Severity severity = Severity::kWARNING);

    bool build();
    bool infer(const size_t& num_test, const size_t& batch_size, const int batch_idx);

private:
    std::shared_ptr<nvinfer1::ICudaEngine mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext mContext;
    samplesCommon::BufferManager mBufferManager;
    bool constructNetwork(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index);
}
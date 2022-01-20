#include "argsParser.h"
#include "buffers.h"
// #include "bertbuffers.h"
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
#include <algorithm>
// #include <torch/torch.h>
#include "../cuda_func/check_exit.cuh"

#include "Python.h"


class Profiler {
    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    using Severity = nvinfer1::ILogger::Severity;

public:
    Profiler(
      const size_t& batch_size_s1 = 1,
      const size_t& batch_size_s2 = 1,
      const int batch_num = 1,
      const int begin_num = 1,
      const Severity severity = Severity::kWARNING);


    bool build(std::vector<std::string> model_name, int batch_size, int engine_per_stage);
    bool build_s0(std::string model_name);
    bool build_s1(std::string model_name);
    bool build_s2(std::string model_name);
    int batch_num_;
    int begin_point_;
    size_t batch_size_s1_;
    size_t batch_size_s2_;
    // void generate_copy_list();
    std::vector<float> infer(const bool separate_or_not, const size_t& num_test,
                             const int batch_idx, const int copy_method, const bool overload, std::string model_name);
    std::vector<float> execute_2stage(const bool separate_or_not, const size_t& num_test,
                             const std::vector<int> record_batch_size, const int copy_method, const bool overload, std::string model_name);
    std::vector<float> execute_multi_stage(const bool separate_or_not, const size_t& num_test,
                             const std::vector<std::vector<int>> record_batch_size, const int copy_method, const bool overload, std::string model_name);
    std::vector<float> bert_execute(const bool separate_or_not, const size_t& num_test,
                             const std::vector<int> record_batch_size, const int copy_method, const bool overload, std::string model_name);
    std::vector<float> bert_execute_multi_stage(const bool separate_or_not, const size_t& num_test,
                             const std::vector<std::vector<int>> record_batch_size, const int copy_method, const bool overload, std::string model_name);

private:
    nvinfer1::DataType model_dtype_;
    bool fp16_{false};
    bool int8_{false};

    cudaStream_t stream_0;
    cudaStream_t stream_1;
    cudaStream_t stream_2;
    cudaEvent_t infer_start;
    cudaEvent_t s1_end;
    cudaEvent_t s2_end;
    std::vector<cudaEvent_t> batch_start;
    std::vector<cudaEvent_t> batch_end;
    std::vector<cudaEvent_t> batch_exit;
    std::vector<cudaEvent_t> batch_exit_2;
    cudaEvent_t check_start;
    cudaEvent_t check_end;
    std::vector<nvinfer1::Dims> input_dims;
    nvinfer1::Dims input_dims_s0;
    nvinfer1::Dims input_dims_s1;
    nvinfer1::Dims input_dims_s2;
    // The input tensor shapes of the second stage of bert are different, so we need two tensors
    nvinfer1::Dims input_dims_s2_0;
    nvinfer1::Dims input_dims_s2_1;
    nvinfer1::Dims output_dims_s0;
    nvinfer1::Dims output_dims_s1;
    nvinfer1::Dims output_dims_s2;
    std::string input_tensor_names_;
    std::string input_tensor_names_0;
    std::string input_tensor_names_1;
    std::string input_tensor_names_2;
    std::string output_tensor_names_;
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> mEngine_list;
    std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> mContext_list;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_s0;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_s1;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_s2;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext_s0;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext_s1;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext_s2;
    // samplesCommon::BufferManager mBufferManager;
    bool construct_s0(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name);
    bool construct_s1(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name);
    bool construct_s2(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name);
    bool construct(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name, int min_bs, int opt_bs, int max_bs);
};
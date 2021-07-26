#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "parserOnnxConfig.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>


class ProfilerConfig {
protected:
    size_t seg_num_{0};
    size_t ee_num_{0};
    std::vector<std::string> segment_fnames_;
    std::vector<std::string> ee_fnames_;
    std::string data_dir_{};

    nvinfer1::DataType model_dtype_;
    bool fp16_{false};
    bool int8_{false};

    std::vector<std::string> input_tensor_names_;
    std::vector<std::string> output_tensor_names_;

public:
    ProfilerConfig() {}
    //!
    //! \brief Construct a new Model Config object
    //!
    //! \param seg_num
    //! \param data_dir
    //! \param model_dtype
    //! \param fp16
    //! \param int8
    //!
    ProfilerConfig(
        const size_t& seg_num, const size_t& ee_num, const std::string& data_dir,
        nvinfer1::DataType model_dtype = nvinfer1::DataType::kFLOAT,
        bool fp16 = true, bool int8 = false)
        : seg_num_(seg_num), ee_num_(ee_num), data_dir_(data_dir), model_dtype_(model_dtype), fp16_(fp16), int8_(int8)
    {
        segment_fnames_.resize(seg_num_);
        ee_fnames_.resize(ee_num_);
    }

    ~ProfilerConfig() {}

public:
    void setModelDtype(const nvinfer1::DataType modelDtype)
    {
        model_dtype_ = modelDtype;
    }

    nvinfer1::DataType getModelDtype() const { return model_dtype_; }

    void setFp16(bool fp16) { fp16_ = fp16; }

    void setInt8(bool int8) { int8_ = int8; }

    bool fp16() { return fp16_; }

    bool int8() { return int8_; }

    void setSegNum(const size_t& seg_num)
    {
        seg_num_ = seg_num;
        segment_fnames_.resize(seg_num_);
    }

    size_t getSegNum() { return seg_num_; }

    void seteeNum(const size_t& ee_num)
    {
        ee_num_ = ee_num;
        ee_fnames_.resize(ee_num_);
    }

    size_t geteeNum() { return ee_num_; }

    void setDataDir(const std::string& data_dir)
    {
        data_dir_ = data_dir_ + "/" + data_dir + "/" + data_dir + "/1/";
    }

    const std::string getDataDir() const { return data_dir_; }

    const std::string getSegFileName(const size_t& index) const
    {
        assert(index < segment_fnames_.size());
        return segment_fnames_[index];
    }
    void setSegFileName(const size_t& index, const std::string& onnxFilename)
    {
        assert(index < segment_fnames_.size());
        segment_fnames_[index] = onnxFilename;
    }
    void setSegFileName(const size_t& index, const char* onnxFilename)
    {
        assert(index < segment_fnames_.size());
        segment_fnames_[index] = std::string(onnxFilename);
    }
    //TODO: seteeFileName
};

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

    bool multistream_infer(
    const size_t& num_test, const size_t& batch_size = 1,
    const size_t& stream_cnt = 1);

    bool subInfer(const size_t& sub_index, cudaStream_t stream);

    bool exportTrtModel(std::string save_path);
    int batch_num_;
    std::vector<float> accuracy;

private:
    ProfilerConfig profiler_config_;
    size_t min_batch_size_;
    size_t opt_batch_size_;
    size_t max_batch_size_;
    //std::vector<cudaStream_t> stream_;
    //std::vector<cudaEvent_t> ms_stop_;
    //std::vector<cudaEvent_t> ee_stop_;
    cudaStream_t stream_[2];
    cudaEvent_t ms_stop_[11];
    cudaEvent_t ee_stop_[6];
    cudaEvent_t start_;
    cudaEvent_t stop_;
    // std::vector<cudaEvent_t> events_;
    template <typename T>
    using DualVector = std::vector<std::vector<T>>;
    nvinfer1::Dims input_dim_;
    std::string input_tensor_name_;
    nvinfer1::Dims output_dim_;
    std::string output_tensor_name_;

    std::vector<nvinfer1::Dims> sub_input_dims_;
    std::vector<nvinfer1::Dims> sub_output_dims_;
    std::vector<std::string> sub_input_tensor_names_;
    std::vector<std::string> sub_output_tensor_names_;
    
    std::vector<nvinfer1::Dims> ee_input_dims_;
    std::vector<nvinfer1::Dims> ee_output_dims_;
    std::vector<std::string> ee_input_tensor_names_;
    std::vector<std::string> ee_output_tensor_names_;

    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> sub_engines_;
    std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> sub_contexts_;
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> ee_engines_;
    std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> ee_contexts_;
    std::vector<samplesCommon::BufferManager> sub_buffer_manager_;
    std::vector<samplesCommon::BufferManager> ee_buffer_manager_;
    std::vector<std::vector<bool>> ee_indicator;

    std::vector<size_t> ee_batch_size;
    std::vector<size_t> sub_batch_size;
    std::vector<int> stage_type{0, 1, 2, 1, 2, 1, 1, 2, 1, 1, 3};
    std::map<int, int> subToEE;

    std::vector<uint8_t> cifarbinary;

    //Logger gLogger_;

    bool constructSubNet(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index);

    bool constructeeNet(
        TRTUniquePtr<nvinfer1::IBuilder>& builder,
        TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
        TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
        TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index);

    std::vector<void*> getDeviceBindings(const size_t& sub_index);
    bool readData();
    bool processInput(const samplesCommon::BufferManager& buffer, const int batch_idx);
    float verifyOutput(const samplesCommon::BufferManager& buffer, const int batch_idx);
    bool controller(const int stage_idx, const int ee_idx);

    bool setBindingDimentions(const size_t& batch_size);

    bool exportSubTrtModel(
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        const std::string& sub_model_fname);

    bool run();
};

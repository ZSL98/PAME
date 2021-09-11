/*
Exit placement
*/

#include <cuda_runtime.h>
#include "exit_placement.h"

Profiler::Profiler(
const size_t& batch_size_s1, const size_t& batch_size_s2,
 const int batch_num, const Severity severity)
: batch_size_s1_(batch_size_s1), batch_size_s2_(batch_size_s2), batch_num_(batch_num)
{
    cudaStreamCreate(&(stream_));
    CUDACHECK(cudaEventCreate(&infer_start));
    CUDACHECK(cudaEventCreate(&s1_end));
    CUDACHECK(cudaEventCreate(&s2_end));
    sample::gLogger.setReportableSeverity(severity);
}

bool Profiler::build_s1()
{
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        std::cout << "Failed to create S1 builder";
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cout << "Failed to create S1 network";
        return false;
    }
    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        std::cout << "Failed to create S1 parser";
        return false;
    }

    auto constructed = construct_s1(builder, network, config, parser);
    if (!constructed) {
        std::cout << "Failed to construct S1 network";
        return false;
    }

    mEngine_s1 = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine_s1) {
        std::cout << "Failed to create S1 engine";
        return false;
    }

    mContext_s1 = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine_s1->createExecutionContext(), samplesCommon::InferDeleter());
}

bool Profiler::build_s2()
{
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        std::cout << "Failed to create S2 builder";
        return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cout << "Failed to create S2 network";
        return false;
    }

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        std::cout << "Failed to create S2 parser";
        return false;
    }

    auto constructed = construct_s2(builder, network, config, parser);
    if (!constructed) {
        std::cout << "Failed to construct S2 network";
        return false;
    }

    mEngine_s2 = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine_s2) {
        std::cout << "Failed to create S2 engine";
        return false;
    }

    mContext_s2 = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine_s2->createExecutionContext(), samplesCommon::InferDeleter());
}

bool Profiler::construct_s1(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/src/exit_placement");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile("resnet_s1.onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    input_dims_s1 = network->getInput(0)->getDimensions();
    input_tensor_names_ = network->getInput(0)->getName();

    nvinfer1::Dims min_dims = input_dims_s1;
    min_dims.d[0] = batch_size_s1_;
    nvinfer1::Dims opt_dims = input_dims_s1;
    opt_dims.d[0] = batch_size_s1_;
    nvinfer1::Dims max_dims = input_dims_s1;
    max_dims.d[0] = batch_size_s1_;

    profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(3_GiB);
    return true;
}

bool Profiler::construct_s2(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/src/exit_placement");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile("resnet_s2.onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    input_dims_s2 = network->getInput(0)->getDimensions();
    input_tensor_names_ = network->getInput(0)->getName();

    nvinfer1::Dims min_dims = input_dims_s2;
    min_dims.d[0] = batch_size_s2_;
    nvinfer1::Dims opt_dims = input_dims_s2;
    opt_dims.d[0] = batch_size_s2_;
    nvinfer1::Dims max_dims = input_dims_s2;
    max_dims.d[0] = batch_size_s1_;

    profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(3_GiB);
    return true;
}

bool Profiler::infer(const size_t& num_test, const int batch_idx, const int copy_method)
{
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(infer_start, stream_));
    input_dims_s1.d[0] = batch_size_s1_;
    mContext_s1->setBindingDimensions(0, input_dims_s1);
    samplesCommon::BufferManager buffer_s1(mEngine_s1, batch_size_s1_);
    buffer_s1.copyInputToDevice();
    auto status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_, nullptr);
    if (!status_s1) {
        std::cout << "Error when inference S1 model" << std::endl;
        return false;
    }
    CUDACHECK(cudaEventRecord(s1_end, stream_));

    std::vector<int> full_copy_list;
    for(int i = 0; i < batch_size_s1_; ++i)
    {
        full_copy_list.push_back(i);
    }
    random_shuffle(full_copy_list.begin(), full_copy_list.end());
    std::vector<int> copy_list;
    for(int i = 0; i < batch_size_s2_; ++i)
    {
        copy_list.push_back(full_copy_list[i]);
    }

    input_dims_s2.d[0] = batch_size_s2_;
    mContext_s2->setBindingDimensions(0, input_dims_s2);
    std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = buffer_s1.getImmediateBuffer(2);
    samplesCommon::BufferManager buffer_s2(mEngine_s2, batch_size_s1_,
                             srcPtr, &copy_list, copy_method);

    auto status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_, nullptr);
    if (!status_s2) {
        std::cout << "Error when inference S2 model" << std::endl;
        return false;
    }
    CUDACHECK(cudaEventRecord(s2_end, stream_));
    CUDACHECK(cudaEventSynchronize(s2_end));
    
    float total_elapsed_time = 0;
    CUDACHECK(cudaEventElapsedTime(&total_elapsed_time, infer_start, s2_end));
    std::cout << "Total elapsed time: " << total_elapsed_time << std::endl;
}

int main(int argc, char** argv)
{
    std::string config_path = "/home/slzhang/projects/ETBA/Inference/src/exit_placement/profiler_config.json";
    FILE* config_fp = fopen(config_path.c_str(), "r");
    if(!config_fp){
        std::cout<<"failed to open config.json"<<endl;
        return -1;
    }
    char read_buffer[65536];

    rapidjson::FileReadStream config_fs(
        config_fp, read_buffer, sizeof(read_buffer));
    rapidjson::Document config_doc;
    config_doc.ParseStream(config_fs);
    Profiler inst = Profiler(
        config_doc["bs_s1"].GetUint(),
        config_doc["bs_s2"].GetUint(), 
        config_doc["bs_num"].GetUint(),
        nvinfer1::ILogger::Severity::kINFO);
    
    inst.build_s1();
    inst.build_s2();
    for (int batch_idx = 0; batch_idx < inst.batch_num_; batch_idx++) 
    {
        inst.infer(config_doc["test_iter"].GetUint(),
                    batch_idx, config_doc["copy_method"].GetUint());
    }

    return 0;
}
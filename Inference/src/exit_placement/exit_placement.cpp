/*
Exit placement
*/

#include <cuda_runtime.h>
#include "exit_placement.h"

Profiler::Profiler(
const size_t& batch_size_s1, const size_t& batch_size_s2,
 const int batch_num, const int begin_point, const Severity severity)
: batch_size_s1_(batch_size_s1), batch_size_s2_(batch_size_s2), begin_point_(begin_point), batch_num_(batch_num)
{
    cudaStreamCreateWithPriority(&(stream_0), cudaStreamDefault, -3);
    cudaStreamCreateWithPriority(&(stream_1), cudaStreamDefault, -2);
    cudaStreamCreateWithPriority(&(stream_2), cudaStreamDefault, -1);
    // cudaStreamCreate(&(stream_0));
    // cudaStreamCreate(&(stream_1));
    // cudaStreamCreate(&(stream_2));
    CUDACHECK(cudaEventCreate(&infer_start));
    CUDACHECK(cudaEventCreate(&s1_end));
    CUDACHECK(cudaEventCreate(&s2_end));
    // CUDACHECK(cudaEventCreate(&check_start));
    // CUDACHECK(cudaEventCreate(&check_end));
    sample::gLogger.setReportableSeverity(severity);
}

bool Profiler::build_s0(std::string model_name)
{
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        std::cout << "Failed to create S0 builder";
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cout << "Failed to create S0 network";
        return false;
    }
    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        std::cout << "Failed to create S0 parser";
        return false;
    }

    auto constructed = construct_s0(builder, network, config, parser, model_name);
    if (!constructed) {
        std::cout << "Failed to construct S0 network";
        return false;
    }

    mEngine_s0 = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine_s0) {
        std::cout << "Failed to create S0 engine";
        return false;
    }

    mContext_s0 = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine_s0->createExecutionContext(), samplesCommon::InferDeleter());
    return true;
}

bool Profiler::build_s1(std::string model_name)
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

    auto constructed = construct_s1(builder, network, config, parser, model_name);
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
    return true;
}

bool Profiler::build_s2(std::string model_name)
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

    auto constructed = construct_s2(builder, network, config, parser, model_name);
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
    return true;
}

bool Profiler::construct_s0(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/src/exit_placement/models");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile(model_name + "_s0.onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    if (model_name == "bert"){
        if (begin_point_ == 0){
            input_dims_s0 = network->getInput(0)->getDimensions();
            size_t sequence_length = 64;

            nvinfer1::Dims min_dims = input_dims_s0;
            min_dims.d[0] = batch_size_s1_;
            min_dims.d[1] = sequence_length;
            nvinfer1::Dims opt_dims = input_dims_s0;
            opt_dims.d[0] = batch_size_s1_;
            opt_dims.d[1] = sequence_length;
            nvinfer1::Dims max_dims = input_dims_s0;
            max_dims.d[0] = batch_size_s1_;
            max_dims.d[1] = sequence_length;

            input_tensor_names_0 = network->getInput(0)->getName();
            input_tensor_names_1 = network->getInput(1)->getName();
            input_tensor_names_2 = network->getInput(2)->getName();

            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

            profile->setDimensions(input_tensor_names_2.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_tensor_names_2.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_tensor_names_2.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
        }
        else {
            input_dims_s2_0 = network->getInput(0)->getDimensions();
            input_dims_s2_1 = network->getInput(1)->getDimensions();
            size_t sequence_length = 64;

            nvinfer1::Dims min_dims_0 = input_dims_s2_0;
            min_dims_0.d[0] = batch_size_s1_;
            min_dims_0.d[1] = sequence_length;
            nvinfer1::Dims opt_dims_0 = input_dims_s2_0;
            opt_dims_0.d[0] = batch_size_s1_;
            opt_dims_0.d[1] = sequence_length;
            nvinfer1::Dims max_dims_0 = input_dims_s2_0;
            max_dims_0.d[0] = batch_size_s1_;
            max_dims_0.d[1] = sequence_length;

            nvinfer1::Dims min_dims_1 = input_dims_s2_1;
            min_dims_1.d[0] = batch_size_s1_;
            min_dims_1.d[1] = sequence_length;
            nvinfer1::Dims opt_dims_1 = input_dims_s2_1;
            opt_dims_1.d[0] = batch_size_s1_;
            opt_dims_1.d[1] = sequence_length;
            nvinfer1::Dims max_dims_1 = input_dims_s2_1;
            max_dims_1.d[0] = batch_size_s1_;
            max_dims_1.d[1] = sequence_length;

            input_tensor_names_0 = network->getInput(0)->getName();
            input_tensor_names_1 = network->getInput(1)->getName();

            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims_0);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims_0);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims_0);

            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims_1);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims_1);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims_1);
        }
    }
    else {
        input_dims_s0 = network->getInput(0)->getDimensions();
        input_tensor_names_ = network->getInput(0)->getName();

        nvinfer1::Dims min_dims = input_dims_s0;
        min_dims.d[0] = batch_size_s1_;
        nvinfer1::Dims opt_dims = input_dims_s0;
        opt_dims.d[0] = batch_size_s1_;
        nvinfer1::Dims max_dims = input_dims_s0;
        max_dims.d[0] = batch_size_s1_;

        profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    }

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(10_GiB);
    return true;
}

bool Profiler::construct_s1(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/src/exit_placement/models");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile(model_name + "_s1.onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    if (model_name == "bert"){
        if (begin_point_ == 0){
            input_dims_s1 = network->getInput(0)->getDimensions();
            size_t sequence_length = 64;

            nvinfer1::Dims min_dims = input_dims_s1;
            min_dims.d[0] = batch_size_s1_;
            min_dims.d[1] = sequence_length;
            nvinfer1::Dims opt_dims = input_dims_s1;
            opt_dims.d[0] = batch_size_s1_;
            opt_dims.d[1] = sequence_length;
            nvinfer1::Dims max_dims = input_dims_s1;
            max_dims.d[0] = batch_size_s1_;
            max_dims.d[1] = sequence_length;

            input_tensor_names_0 = network->getInput(0)->getName();
            input_tensor_names_1 = network->getInput(1)->getName();
            input_tensor_names_2 = network->getInput(2)->getName();

            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

            profile->setDimensions(input_tensor_names_2.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_tensor_names_2.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_tensor_names_2.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
        }
        else {
            input_dims_s2_0 = network->getInput(0)->getDimensions();
            input_dims_s2_1 = network->getInput(1)->getDimensions();
            size_t sequence_length = 64;

            nvinfer1::Dims min_dims_0 = input_dims_s2_0;
            min_dims_0.d[0] = batch_size_s1_;
            min_dims_0.d[1] = sequence_length;
            nvinfer1::Dims opt_dims_0 = input_dims_s2_0;
            opt_dims_0.d[0] = batch_size_s1_;
            opt_dims_0.d[1] = sequence_length;
            nvinfer1::Dims max_dims_0 = input_dims_s2_0;
            max_dims_0.d[0] = batch_size_s1_;
            max_dims_0.d[1] = sequence_length;

            nvinfer1::Dims min_dims_1 = input_dims_s2_1;
            min_dims_1.d[0] = batch_size_s1_;
            min_dims_1.d[1] = sequence_length;
            nvinfer1::Dims opt_dims_1 = input_dims_s2_1;
            opt_dims_1.d[0] = batch_size_s1_;
            opt_dims_1.d[1] = sequence_length;
            nvinfer1::Dims max_dims_1 = input_dims_s2_1;
            max_dims_1.d[0] = batch_size_s1_;
            max_dims_1.d[1] = sequence_length;

            input_tensor_names_0 = network->getInput(0)->getName();
            input_tensor_names_1 = network->getInput(1)->getName();

            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims_0);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims_0);
            profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims_0);

            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims_1);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims_1);
            profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims_1);
        }
    }
    else {
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
    }

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(10_GiB);
    // config->setMinTimingIterations(2);
    // config->setAvgTimingIterations(1);
    // std::cout << "getMinTimingIterations: " << config->getMinTimingIterations() << std::endl;
    // std::cout << "getAvgTimingIterations: " << config->getAvgTimingIterations() << std::endl;
    return true;
}

bool Profiler::construct_s2(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/src/exit_placement/models");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile(model_name + "_s2.onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    if (model_name == "bert"){
        input_dims_s2_0 = network->getInput(0)->getDimensions();
        input_dims_s2_1 = network->getInput(1)->getDimensions();
        size_t sequence_length = 64;

        nvinfer1::Dims min_dims_0 = input_dims_s2_0;
        min_dims_0.d[0] = batch_size_s2_;
        min_dims_0.d[1] = sequence_length;
        nvinfer1::Dims opt_dims_0 = input_dims_s2_0;
        opt_dims_0.d[0] = batch_size_s1_;
        opt_dims_0.d[1] = sequence_length;
        nvinfer1::Dims max_dims_0 = input_dims_s2_0;
        max_dims_0.d[0] = batch_size_s1_;
        max_dims_0.d[1] = sequence_length;

        nvinfer1::Dims min_dims_1 = input_dims_s2_1;
        min_dims_1.d[0] = batch_size_s2_;
        min_dims_1.d[1] = sequence_length;
        nvinfer1::Dims opt_dims_1 = input_dims_s2_1;
        opt_dims_1.d[0] = batch_size_s1_;
        opt_dims_1.d[1] = sequence_length;
        nvinfer1::Dims max_dims_1 = input_dims_s2_1;
        max_dims_1.d[0] = batch_size_s1_;
        max_dims_1.d[1] = sequence_length;

        input_tensor_names_0 = network->getInput(0)->getName();
        input_tensor_names_1 = network->getInput(1)->getName();

        profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims_0);
        profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims_0);
        profile->setDimensions(input_tensor_names_0.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims_0);

        profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims_1);
        profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims_1);
        profile->setDimensions(input_tensor_names_1.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims_1);

    }
    else{
        input_dims_s2 = network->getInput(0)->getDimensions();
        input_tensor_names_ = network->getInput(0)->getName();
        // std::cout << network->getInput(0)->getName() << std::endl;
        // std::cout << network->getInput(1)->getName() << std::endl;

        nvinfer1::Dims min_dims = input_dims_s2;
        min_dims.d[0] = batch_size_s2_;
        nvinfer1::Dims opt_dims = input_dims_s2;
        opt_dims.d[0] = batch_size_s1_;
        nvinfer1::Dims max_dims = input_dims_s2;
        max_dims.d[0] = batch_size_s1_;

        profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input_tensor_names_.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    }
    // config->setMinTimingIterations(2);
    // config->setAvgTimingIterations(1);
    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(10_GiB);
    return true;
}

std::vector<float> Profiler::bert_execute(const bool separate_or_not, const size_t& num_test,
                 const int batch_idx, const int copy_method, const bool overload, std::string model_name)
{
    float elapsed_time = 0;
    std::vector<float> metrics;
    if (separate_or_not) {

        size_t sequence_length = 64;

        if (begin_point_ == 0) {
            input_dims_s1.d[0] = batch_size_s1_;
            input_dims_s1.d[1] = sequence_length;
            mContext_s1->setBindingDimensions(0, input_dims_s1);
            mContext_s1->setBindingDimensions(1, input_dims_s1);
            mContext_s1->setBindingDimensions(2, input_dims_s1);
        }
        else {
            input_dims_s2_0.d[0] = batch_size_s1_;
            input_dims_s2_1.d[0] = batch_size_s1_;
            input_dims_s2_0.d[1] = sequence_length;
            input_dims_s2_1.d[1] = sequence_length;
            mContext_s1->setBindingDimensions(0, input_dims_s2_0);
            mContext_s1->setBindingDimensions(1, input_dims_s2_1);
        }

        samplesCommon::BertBufferManager buffer_s1(mEngine_s1, batch_size_s1_);
        buffer_s1.copyInputToDeviceAsync(stream_1);
        auto status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
        if (!status_s1) {
            std::cout << "Error when inferring S1 model" << std::endl;
        }

        std::shared_ptr<samplesCommon::BertManagedBuffer> exitPtr;
        if (begin_point_ == 0) {
            exitPtr = buffer_s1.getImmediateBuffer(4);
        }
        else {
            exitPtr = buffer_s1.getImmediateBuffer(3);
        }

        float* exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
        int *copy_list;
        int size = (int) batch_size_s1_*sizeof(int);
        cudaMalloc(&copy_list, size);
        // max_reduction_r(exitPtr_device, copy_list, stream_1);

        int next_batch_size = batch_size_s2_;
        int *fake_copy_list;
        cudaMalloc(&fake_copy_list, (int) next_batch_size*sizeof(int));

        // Generate on the CPU side and copy to the GPU
        std::vector<int> full_copy_list;
        for(int i = 0; i < batch_size_s1_; ++i)
        {
            full_copy_list.push_back(i);
        }
        random_shuffle(full_copy_list.begin(), full_copy_list.end());
        int fake_copy_list_host[next_batch_size];
        for(int i = 0; i < next_batch_size ; ++i){
            fake_copy_list_host[i] = full_copy_list[i];
        }
        cudaMemcpy(fake_copy_list, fake_copy_list_host, (int) next_batch_size*sizeof(int), cudaMemcpyHostToDevice);

        input_dims_s2_0.d[0] = next_batch_size;
        input_dims_s2_1.d[0] = next_batch_size;
        input_dims_s2_0.d[1] = sequence_length;
        input_dims_s2_1.d[1] = sequence_length;
        mContext_s2->setBindingDimensions(0, input_dims_s2_0);
        mContext_s2->setBindingDimensions(1, input_dims_s2_1);

        std::shared_ptr<samplesCommon::BertManagedBuffer> srcPtr;
        if (begin_point_ == 0) {
            srcPtr = buffer_s1.getImmediateBuffer(3);
        }
        else {
            srcPtr = buffer_s1.getImmediateBuffer(2);
        }

        samplesCommon::BertBufferManager buffer_s2(mEngine_s2, batch_size_s1_,
                                srcPtr, fake_copy_list, next_batch_size, copy_method);

        auto status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
        if (!status_s2) {
            std::cout << "Error when inferring S2 model" << std::endl;
        }

        CUDACHECK(cudaDeviceSynchronize());

        CUDACHECK(cudaEventRecord(infer_start, stream_1));
        for (int i = 0; i < batch_num_; i++){
            // buffer_s1.copyInputToDeviceAsync(stream_2);
            // CUDACHECK(cudaDeviceSynchronize());

            status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
            if (!status_s1) {
                std::cout << "Error when inferring S1 model" << std::endl;
            }

            if (next_batch_size == 0) {
                CUDACHECK(cudaDeviceSynchronize());
                continue;
            }

            if (begin_point_ == 0) {
                exitPtr = buffer_s1.getImmediateBuffer(4);
            }
            else {
                exitPtr = buffer_s1.getImmediateBuffer(3);
            }
            exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
            // TODO: WARNING
            // max_reduction_r(exitPtr_device, copy_list, stream_1);

            std::shared_ptr<samplesCommon::BertManagedBuffer> manBuf_ptr;
            if (begin_point_ == 0) {
                manBuf_ptr = buffer_s1.getImmediateBuffer(3);
            }
            else {
                manBuf_ptr = buffer_s1.getImmediateBuffer(2);
            }
            std::shared_ptr<samplesCommon::BertManagedBuffer> new_manBuf = buffer_s2.getImmediateBuffer(0);
            float* dstPtr_ = static_cast<float*>(new_manBuf->deviceBuffer.data());
            float* srcPtr_ = static_cast<float*>(manBuf_ptr->deviceBuffer.data());
            auto dims = mEngine_s2->getBindingDimensions(0);
            dims.d[0] = 1;
            size_t singleVol = samplesCommon::volume(dims);

            buffercopy(dstPtr_, srcPtr_, singleVol*next_batch_size, fake_copy_list, singleVol, stream_1);

            status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
            if (!status_s2) {
                std::cout << "Error when inferring S2 model" << std::endl;
            }

            CUDACHECK(cudaDeviceSynchronize());
        }

        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventRecord(s2_end, stream_0));
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventElapsedTime(&elapsed_time, infer_start, s2_end));
        metrics.push_back(elapsed_time);
    }
    else {
        CUDACHECK(cudaDeviceSynchronize());
        size_t sequence_length = 64;
        if (begin_point_ == 0) {
            input_dims_s0.d[0] = batch_size_s1_;
            input_dims_s0.d[1] = sequence_length;
            mContext_s0->setBindingDimensions(0, input_dims_s0);
            mContext_s0->setBindingDimensions(1, input_dims_s0);
            mContext_s0->setBindingDimensions(2, input_dims_s0);
        }
        else {
            input_dims_s2_0.d[0] = batch_size_s1_;
            input_dims_s2_1.d[0] = batch_size_s1_;
            input_dims_s2_0.d[1] = sequence_length;
            input_dims_s2_1.d[1] = sequence_length;
            mContext_s0->setBindingDimensions(0, input_dims_s2_0);
            mContext_s0->setBindingDimensions(1, input_dims_s2_1);
        }

        samplesCommon::BertBufferManager buffer_s0(mEngine_s0, batch_size_s1_);
        buffer_s0.copyInputToDeviceAsync(stream_0);
        auto status_s0 = mContext_s0->enqueueV2(buffer_s0.getDeviceBindings().data(), stream_0, nullptr);
        if (!status_s0) {
            std::cout << "Error when inferring the full model" << std::endl;
        }
        CUDACHECK(cudaDeviceSynchronize());

        CUDACHECK(cudaEventRecord(infer_start, stream_0));
        for (int i = 0; i < batch_num_; i++){
            status_s0 = mContext_s0->enqueueV2(buffer_s0.getDeviceBindings().data(), stream_0, nullptr);
            if (!status_s0) {
                std::cout << "Error when inferring the full model" << std::endl;
            }
            CUDACHECK(cudaDeviceSynchronize());
        }
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventRecord(s2_end, stream_0));
        CUDACHECK(cudaEventSynchronize(s2_end));
    
        CUDACHECK(cudaEventElapsedTime(&elapsed_time, infer_start, s2_end));
        metrics.push_back(elapsed_time);
    }
    return metrics;
}

std::vector<float> Profiler::execute(const bool separate_or_not, const size_t& num_test,
                 const int batch_idx, const int copy_method, const bool overload, std::string model_name)
{
    float elapsed_time = 0;
    std::vector<float> metrics;
    if (separate_or_not) {
        input_dims_s1.d[0] = batch_size_s1_;
        mContext_s1->setBindingDimensions(0, input_dims_s1);
        samplesCommon::BufferManager buffer_s1(mEngine_s1, batch_size_s1_);
        buffer_s1.copyInputToDeviceAsync(stream_1);
        auto status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
        if (!status_s1) {
            std::cout << "Error when inferring S1 model" << std::endl;
        }

        std::shared_ptr<samplesCommon::ManagedBuffer> exitPtr = buffer_s1.getImmediateBuffer(2);
            
        float* exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
        int *copy_list;
        int size = (int) batch_size_s1_*sizeof(int);
        cudaMalloc(&copy_list, size);
        max_reduction_r(exitPtr_device, copy_list, stream_1);

        int next_batch_size = batch_size_s2_;
        int *fake_copy_list;
        cudaMalloc(&fake_copy_list, (int) next_batch_size*sizeof(int));

        // Generate on the CPU side and copy to the GPU
        std::vector<int> full_copy_list;
        for(int i = 0; i < batch_size_s1_; ++i)
        {
            full_copy_list.push_back(i);
        }
        random_shuffle(full_copy_list.begin(), full_copy_list.end());
        int fake_copy_list_host[next_batch_size];
        for(int i = 0; i < next_batch_size ; ++i){
            fake_copy_list_host[i] = full_copy_list[i];
        }
        cudaMemcpy(fake_copy_list, fake_copy_list_host, (int) next_batch_size*sizeof(int), cudaMemcpyHostToDevice);

        input_dims_s2.d[0] = next_batch_size;
        mContext_s2->setBindingDimensions(0, input_dims_s2);

        std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = buffer_s1.getImmediateBuffer(1);
        samplesCommon::BufferManager buffer_s2(mEngine_s2, batch_size_s1_,
                                srcPtr, fake_copy_list, next_batch_size, copy_method);

        auto status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
        if (!status_s2) {
            std::cout << "Error when inferring S2 model" << std::endl;
        }

        CUDACHECK(cudaDeviceSynchronize());

        CUDACHECK(cudaEventRecord(infer_start, stream_1));
        for (int i = 0; i < batch_num_; i++){
            // buffer_s1.copyInputToDeviceAsync(stream_2);
            // CUDACHECK(cudaDeviceSynchronize());

            status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
            if (!status_s1) {
                std::cout << "Error when inferring S1 model" << std::endl;
            }

            if (next_batch_size == 0) {
                CUDACHECK(cudaDeviceSynchronize());
                continue;
            }

            exitPtr = buffer_s1.getImmediateBuffer(2);
            exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
            max_reduction_p(exitPtr_device, copy_list, stream_1);


            std::shared_ptr<samplesCommon::ManagedBuffer> manBuf_ptr = buffer_s1.getImmediateBuffer(1);
            std::shared_ptr<samplesCommon::ManagedBuffer> new_manBuf = buffer_s2.getImmediateBuffer(0);
            float* dstPtr_ = static_cast<float*>(new_manBuf->deviceBuffer.data());
            float* srcPtr_ = static_cast<float*>(manBuf_ptr->deviceBuffer.data());
            auto dims = mEngine_s2->getBindingDimensions(0);
            dims.d[0] = 1;
            size_t singleVol = samplesCommon::volume(dims);

            buffercopy(dstPtr_, srcPtr_, singleVol*next_batch_size, fake_copy_list, singleVol, stream_1);

            // nvinfer1::DataType type = mEngine_s1->getBindingDataType(1);
            // size_t tSize = samplesCommon::getElementSize(type);
            // for (int stepOveridx = 0; stepOveridx < next_batch_size; stepOveridx++) {
            //     const cudaMemcpyKind memcpyType = cudaMemcpyDeviceToDevice;
            //     cudaMemcpy(dstPtr_ + stepOveridx*singleVol, 
            //                     srcPtr_ + fake_copy_list_host[stepOveridx]*singleVol, 
            //                     singleVol*tSize, memcpyType);
            // }

            status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
            if (!status_s2) {
                std::cout << "Error when inferring S2 model" << std::endl;
            }

            CUDACHECK(cudaDeviceSynchronize());
        }

        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventRecord(s2_end, stream_0));
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventElapsedTime(&elapsed_time, infer_start, s2_end));
        metrics.push_back(elapsed_time);
    }
    else {
        CUDACHECK(cudaDeviceSynchronize());
        input_dims_s0.d[0] = batch_size_s1_;
        mContext_s0->setBindingDimensions(0, input_dims_s0);
        samplesCommon::BufferManager buffer_s0(mEngine_s0, batch_size_s1_);
        buffer_s0.copyInputToDeviceAsync(stream_0);
        auto status_s0 = mContext_s0->enqueueV2(buffer_s0.getDeviceBindings().data(), stream_0, nullptr);
        if (!status_s0) {
            std::cout << "Error when inferring the full model" << std::endl;
        }
        CUDACHECK(cudaDeviceSynchronize());

        CUDACHECK(cudaEventRecord(infer_start, stream_0));
        for (int i = 0; i < batch_num_; i++){
            status_s0 = mContext_s0->enqueueV2(buffer_s0.getDeviceBindings().data(), stream_0, nullptr);
            if (!status_s0) {
                std::cout << "Error when inferring the full model" << std::endl;
            }
            CUDACHECK(cudaDeviceSynchronize());
        }
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventRecord(s2_end, stream_0));
        CUDACHECK(cudaEventSynchronize(s2_end));
    
        CUDACHECK(cudaEventElapsedTime(&elapsed_time, infer_start, s2_end));
        metrics.push_back(elapsed_time);
    }
    return metrics;
}


std::vector<float> Profiler::infer(const bool separate_or_not, const size_t& num_test,
                 const int batch_idx, const int copy_method, const bool overload, std::string model_name)
{
    float elapsed_time = 0;
    float elapsed_time_s1 = 0;
    float elapsed_time_s2 = 0;
    float check_time = 0;
    std::vector<float> metrics;
    if (separate_or_not) {

        CUDACHECK(cudaDeviceSynchronize());
        // CUDACHECK(cudaEventRecord(infer_start, stream_1));

        if (model_name == "bert"){
            size_t sequence_length = 64;
            input_dims_s1.d[0] = batch_size_s1_;
            input_dims_s1.d[1] = sequence_length;
            mContext_s1->setBindingDimensions(0, input_dims_s1);
            mContext_s1->setBindingDimensions(1, input_dims_s1);
            mContext_s1->setBindingDimensions(2, input_dims_s1);
        }
        else {
            input_dims_s1.d[0] = batch_size_s1_;
            mContext_s1->setBindingDimensions(0, input_dims_s1);
        }

        // samplesCommon::BufferManager buffer_s1(mEngine_s1, batch_size_s1_);
        // buffer_s1.copyInputToDeviceAsync(stream_1);
        // auto status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
        // if (!status_s1) {
        //     std::cout << "Error when inferring S1 model" << std::endl;
        // }
        // CUDACHECK(cudaEventSynchronize(s1_end));
        // buffer_s1.copyInputToDeviceAsync(stream_1);
        // CUDACHECK(cudaDeviceSynchronize());


        /* Below is the module for check */
        // On the GPU side
        // std::shared_ptr<samplesCommon::ManagedBuffer> exitPtr = buffer_s1.getImmediateBuffer(1);
        // float* exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());

        // int *copy_list;
        // int size = (int) batch_size_s1_*sizeof(int);
        // cudaMalloc(&copy_list, size);
        // max_reduction_r(exitPtr_device, copy_list, stream_1);


        /* A very inefficient check module on GPU
        CUDACHECK(cudaEventRecord(check_start, stream_0));

        float threshold = 0.5;
        int length = 1000;
        int* copy_list;
        int size = (int) batch_size_s1_*sizeof(int);
        cudaMalloc(&copy_list, size);
        cudaMemset(copy_list, 0, size);
        cls_copy_list(exitPtr_, copy_list, threshold, length, batch_size_s1_);
        CUDACHECK(cudaEventRecord(check_end, stream_0));
        CUDACHECK(cudaEventSynchronize(check_end));
        */

        /* Code for generate the copy list randomly */
        int next_batch_size = batch_size_s2_;
        if (overload) {
            next_batch_size = batch_size_s1_;
        }
        int *fake_copy_list;
        // cudaMalloc(&fake_copy_list, (int) next_batch_size*sizeof(int));

        // Generate on the CPU side and copy to the GPU
        std::vector<int> full_copy_list;
        for(int i = 0; i < batch_size_s1_; ++i)
        {
            full_copy_list.push_back(i);
        }
        random_shuffle(full_copy_list.begin(), full_copy_list.end());
        int fake_copy_list_host[next_batch_size];
        for(int i = 0; i < next_batch_size ; ++i){
            fake_copy_list_host[i] = full_copy_list[i];
        }
        // cudaMemcpy(fake_copy_list, fake_copy_list_host, (int) next_batch_size*sizeof(int), cudaMemcpyHostToDevice);

        // TODO: Generate on the GPU side
        // generate_fake_copy_list(batch_size_s1_, next_batch_size, fake_copy_list);

        if (model_name == "bert"){
            size_t sequence_length = 64;
            input_dims_s2_0.d[0] = next_batch_size;
            input_dims_s2_1.d[0] = next_batch_size;
            input_dims_s2_0.d[1] = sequence_length;
            input_dims_s2_1.d[1] = sequence_length;
            mContext_s2->setBindingDimensions(0, input_dims_s2_0);
            mContext_s2->setBindingDimensions(1, input_dims_s2_1);
        }
        else {
            input_dims_s2.d[0] = next_batch_size;
            mContext_s2->setBindingDimensions(0, input_dims_s2);
        }
        // std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = buffer_s1.getImmediateBuffer(2);
        // std::cout << next_batch_size << std::endl;
        // samplesCommon::BufferManager buffer_s2(mEngine_s2, batch_size_s1_,
        //                         srcPtr, fake_copy_list, next_batch_size, copy_method);

        // auto status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
        // if (!status_s2) {
        //     std::cout << "Error when inferring S2 model" << std::endl;
        // }

        // CUDACHECK(cudaDeviceSynchronize());

        // CUDACHECK(cudaEventRecord(s1_end, stream_1));
        // for (int i = 0; i < 10; i++){
            // buffer_s1.copyInputToDeviceAsync(stream_1);
            // CUDACHECK(cudaDeviceSynchronize());

            // status_s1 = mContext_s1->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
            // if (!status_s1) {
            //     std::cout << "Error when inferring S1 model" << std::endl;
            // }

            // exitPtr = buffer_s1.getImmediateBuffer(1);
            // exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
            // max_reduction_r(exitPtr_device, copy_list, stream_1);

            // std::shared_ptr<samplesCommon::ManagedBuffer> manBuf_ptr = buffer_s1.getImmediateBuffer(2);
            // std::shared_ptr<samplesCommon::ManagedBuffer> new_manBuf = buffer_s2.getImmediateBuffer(0);
            // float* dstPtr_ = static_cast<float*>(new_manBuf->deviceBuffer.data());
            // float* srcPtr_ = static_cast<float*>(manBuf_ptr->deviceBuffer.data());
            // auto dims = mEngine_s2->getBindingDimensions(0);
            // dims.d[0] = 1;
            // size_t singleVol = samplesCommon::volume(dims);

            // buffercopy(dstPtr_, srcPtr_, singleVol*next_batch_size, fake_copy_list, singleVol, stream_1);

            // nvinfer1::DataType type = mEngine_s1->getBindingDataType(1);
            // size_t tSize = samplesCommon::getElementSize(type);
            // for (int stepOveridx = 0; stepOveridx < next_batch_size; stepOveridx++) {
            //     const cudaMemcpyKind memcpyType = cudaMemcpyDeviceToDevice;
            //     cudaMemcpy(dstPtr_ + stepOveridx*singleVol, 
            //                     srcPtr_ + fake_copy_list_host[stepOveridx]*singleVol, 
            //                     singleVol*tSize, memcpyType);
            // }

        //     status_s2 = mContext_s2->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_1, nullptr);
        //     if (!status_s2) {
        //         std::cout << "Error when inferring S2 model" << std::endl;
        //     }

        //     CUDACHECK(cudaDeviceSynchronize());
        // }

        // CUDACHECK(cudaDeviceSynchronize());
        // CUDACHECK(cudaEventRecord(s2_end, stream_1));
        // CUDACHECK(cudaEventSynchronize(s2_end));
        

        // CUDACHECK(cudaEventElapsedTime(&elapsed_time_s1, infer_start, s1_end));
        // CUDACHECK(cudaEventElapsedTime(&elapsed_time_s2, s1_end, s2_end));
        // CUDACHECK(cudaEventElapsedTime(&elapsed_time, infer_start, s2_end));

        // CUDACHECK(cudaEventElapsedTime(&check_time, check_start, check_end));
        
        // metrics.push_back(elapsed_time_s1);
        // metrics.push_back(elapsed_time_s2);
        // metrics.push_back(elapsed_time);

        // metrics.push_back(check_time);
        // std::cout << "Checking exits takes: " << check_time << "ms." << std::endl;
    }
    else {
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaEventRecord(infer_start, stream_0));
        input_dims_s0.d[0] = batch_size_s1_;
        mContext_s0->setBindingDimensions(0, input_dims_s0);
        samplesCommon::BufferManager buffer_s0(mEngine_s0, batch_size_s1_);
        buffer_s0.copyInputToDeviceAsync(stream_0);
        auto status_s0 = mContext_s0->enqueueV2(buffer_s0.getDeviceBindings().data(), stream_0, nullptr);
        if (!status_s0) {
            std::cout << "Error when inferring the full model" << std::endl;
        }

        CUDACHECK(cudaDeviceSynchronize());

        CUDACHECK(cudaEventRecord(s1_end, stream_0));
        for (int i = 0; i < 10; i++){
            status_s0 = mContext_s0->enqueueV2(buffer_s0.getDeviceBindings().data(), stream_0, nullptr);
            if (!status_s0) {
                std::cout << "Error when inferring the full model" << std::endl;
            }
            CUDACHECK(cudaDeviceSynchronize());
        }
        CUDACHECK(cudaEventRecord(s2_end, stream_0));
        CUDACHECK(cudaEventSynchronize(s2_end));
    
        CUDACHECK(cudaEventElapsedTime(&elapsed_time, s1_end, s2_end));
        metrics.push_back(elapsed_time);
    }
    return metrics;
}

bool model_generation(std::string model_name, const int start_point, const int end_point)
{
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/slzhang/projects/ETBA/Inference/src/exit_placement')");
	PyObject* pModule = PyImport_ImportModule("model_export_v2");
	if( pModule == NULL ){
		cout <<"module not found" << endl;
		return 1;
	}
    std::cout << "module found" << std::endl;
	PyObject* pFunc = PyObject_GetAttrString(pModule, "model_export_func");
	if( !pFunc || !PyCallable_Check(pFunc)){
		cout <<"not found function model_export_func" << endl;
		return 0;
	}
    PyObject* args = Py_BuildValue("sii", model_name.c_str(), start_point, end_point);
    PyObject* pRet = PyObject_CallObject(pFunc, args);
    Py_DECREF(args);
    Py_DECREF(pRet);
    return true;
}

int main(int argc, char** argv)
{
    int nGpuId = 0;
    cudaSetDevice(nGpuId);
    // int leastPriority;
    // int greatestPriority;
    // cudaDeviceGetStreamPriorityRange (&leastPriority, &greatestPriority );
    // std::cout << leastPriority << "  " << greatestPriority << std::endl;

    std::string model_name = argv[1];
    std::cout << "Profiling model: " << model_name + "!" << std::endl;
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

    std::ofstream outFile;
    // Py_Initialize();
    int extend_max_block = 1;
    if (config_doc["seperate_or_not"].GetBool()){
        for (int split_point = config_doc["split_point"].GetUint(); split_point < config_doc["termi_point"].GetUint(); split_point=split_point+config_doc["stage_interval"].GetUint())
        {

            bool model_generated = model_generation(model_name, config_doc["begin_point"].GetUint(), split_point);
            if(!model_generated){
                std::cout<<"failed to export models"<<endl;
                return -1;
            }
            
            // std::vector<float> avg_elapsed_time_s1;
            // std::vector<float> avg_elapsed_time_s2;
            std::vector<float> elapsed_time;
            // std::vector<float> avg_elapsed_time_overload;
            int infer_batch_size_s1 = config_doc["bs_s1"].GetUint();
            int batch_interval = config_doc["b_interval"].GetUint();
            for (int infer_batch_size_s2 = 0; infer_batch_size_s2 <= infer_batch_size_s1;
                     infer_batch_size_s2 = infer_batch_size_s2 + batch_interval)
            {
                // size_t infer_batch_size_s2 = config_doc["bs_s2"].GetUint() * batch_scale / 4;
                Profiler inst = Profiler(infer_batch_size_s1, infer_batch_size_s2, 
                                        config_doc["bs_num"].GetUint(), config_doc["begin_point"].GetUint(),
                                        nvinfer1::ILogger::Severity::kERROR);
                inst.build_s1(model_name);
                inst.build_s2(model_name);
                std::vector<float> metrics;
                if (model_name == "bert"){
                    metrics = inst.bert_execute(config_doc["seperate_or_not"].GetBool(),
                                                config_doc["test_iter"].GetUint(), 0, 
                                                config_doc["copy_method"].GetUint(), false, model_name);
                }
                else {
                    metrics = inst.execute(config_doc["seperate_or_not"].GetBool(),
                                            config_doc["test_iter"].GetUint(), 0, 
                                            config_doc["copy_method"].GetUint(), false, model_name);                    
                }

                std::cout << "Batch size: " << infer_batch_size_s2 << "/" << infer_batch_size_s1 << "  Elapsed time: " << metrics[0]/inst.batch_num_ << std::endl;
                // avg_elapsed_time_s1.push_back(total_elapsed_time_s1/inst.batch_num_);
                // avg_elapsed_time_s2.push_back(total_elapsed_time_s2/inst.batch_num_);
                elapsed_time.push_back(metrics[0]/inst.batch_num_);

            }
            
            outFile.open("/home/slzhang/projects/ETBA/Inference/src/exit_placement/results/config_v100_" + model_name + "_" +
                            to_string(config_doc["bs_s1"].GetUint()) + "_l" + to_string(config_doc["begin_point"].GetUint()) + ".csv", ios::app);
            outFile << config_doc["begin_point"].GetUint() << ',' << split_point << ',';

            for (int i = 0; i < elapsed_time.size(); ++i){
                outFile << elapsed_time.at(i) << ',';
                if (i == elapsed_time.size() - 1) {outFile << endl;}
            }

            outFile.close();
        }
    }
    else {
        size_t batch_size_s1 = config_doc["bs_s1"].GetUint();
        size_t batch_size_s2 = config_doc["bs_s2"].GetUint();
        Profiler inst = Profiler(batch_size_s1, batch_size_s2, 
                                config_doc["bs_num"].GetUint(), config_doc["begin_point"].GetUint(),
                                nvinfer1::ILogger::Severity::kERROR);
        inst.build_s0(model_name);
        std::vector<float> metrics;
        if (model_name == "bert"){
            metrics = inst.bert_execute(config_doc["seperate_or_not"].GetBool(),
                                        config_doc["test_iter"].GetUint(), 0, 
                                        config_doc["copy_method"].GetUint(), false, model_name);
        }
        else {
            metrics = inst.execute(config_doc["seperate_or_not"].GetBool(),
                                    config_doc["test_iter"].GetUint(), 0, 
                                    config_doc["copy_method"].GetUint(), false, model_name);                    
        }
        float total_elapsed_time = metrics[0];
        float avg_elapsed_time = total_elapsed_time/inst.batch_num_;
        std::cout << "Average elapsed time of the begin-from-" << config_doc["begin_point"].GetUint() << " model: " 
                    << avg_elapsed_time << std::endl;
    }
    // Py_Finalize();
    return 0;
}
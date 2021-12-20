/*
Exit placement
*/

#include <cuda_runtime.h>
#include "run_engine.h"

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


bool Profiler::build(std::vector<std::string> model_name, int batch_size, int engine_per_stage)
{
    int stage_num = model_name.size();
    // mEngine_list = std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>;
    // mContext_list = std::vector<std::shared_ptr<nvinfer1::IExecutionContext>>;
    for (int i = 0; i < stage_num; i++){
        std::cout << "Building engines for stage " << i << std::endl;
        for (int j = 1; j <= engine_per_stage; j++){

            auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
            if (!builder) {
                std::cout << "Failed to create builder";
                return false;
            }

            const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
            if (!network) {
                std::cout << "Failed to create network";
                return false;
            }

            auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
            if (!config) {
                return false;
            }

            auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
            if (!parser) {
                std::cout << "Failed to create parser";
                return false;
            }

            auto constructed = construct(builder, network, config, parser, model_name[i], 1, batch_size*j/engine_per_stage, batch_size*j/engine_per_stage);
            if (!constructed) {
                std::cout << "Failed to construct network";
                return false;
            }

            mEngine_list.push_back(std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter()));
            mContext_list.push_back(std::shared_ptr<nvinfer1::IExecutionContext>(mEngine_list[i*engine_per_stage+j-1]->createExecutionContext(), samplesCommon::InferDeleter()));
            std::cout << "Building engine for batch size: " << batch_size*j/engine_per_stage << std::endl;
        }
    }
    return true;
}


bool Profiler::construct(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, std::string model_name, int min_bs, int opt_bs, int max_bs)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/src/run_engine/models");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile(model_name + ".onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    if (model_name == "bert"){
        if (begin_point_ == 0){
            input_dims_s2 = network->getInput(0)->getDimensions();
            std::cout << input_dims_s2 << std::endl;
            size_t sequence_length = 7;

            nvinfer1::Dims min_dims = input_dims_s2;
            min_dims.d[0] = min_bs;
            min_dims.d[1] = sequence_length;
            nvinfer1::Dims opt_dims = input_dims_s2;
            opt_dims.d[0] = opt_bs;
            opt_dims.d[1] = sequence_length;
            nvinfer1::Dims max_dims = input_dims_s2;
            max_dims.d[0] = max_bs;
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
            size_t sequence_length = 7;

            nvinfer1::Dims min_dims_0 = input_dims_s2_0;
            min_dims_0.d[0] = min_bs;
            min_dims_0.d[1] = sequence_length;
            nvinfer1::Dims opt_dims_0 = input_dims_s2_0;
            opt_dims_0.d[0] = opt_bs;
            opt_dims_0.d[1] = sequence_length;
            nvinfer1::Dims max_dims_0 = input_dims_s2_0;
            max_dims_0.d[0] = max_bs;
            max_dims_0.d[1] = sequence_length;

            nvinfer1::Dims min_dims_1 = input_dims_s2_1;
            min_dims_1.d[0] = min_bs;
            min_dims_1.d[1] = sequence_length;
            nvinfer1::Dims opt_dims_1 = input_dims_s2_1;
            opt_dims_1.d[0] = opt_bs;
            opt_dims_1.d[1] = sequence_length;
            nvinfer1::Dims max_dims_1 = input_dims_s2_1;
            max_dims_1.d[0] = max_bs;
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
        input_dims.push_back(network->getInput(0)->getDimensions());
        input_tensor_names_ = network->getInput(0)->getName();

        nvinfer1::Dims min_dims = input_dims.back();
        min_dims.d[0] = min_bs;
        nvinfer1::Dims opt_dims = input_dims.back();
        opt_dims.d[0] = opt_bs;
        nvinfer1::Dims max_dims = input_dims.back();
        max_dims.d[0] = max_bs;

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




std::vector<int> generate_copy_list(std::string movon_dict_path)
{
    FILE* config_fp = fopen(movon_dict_path.c_str(), "r");
    if(!config_fp){
        std::cout<<"failed to open config.json"<<endl;
    }
    char read_buffer[65536];

    rapidjson::FileReadStream config_fs(
        config_fp, read_buffer, sizeof(read_buffer));
    rapidjson::Document config_doc;
    config_doc.ParseStream(config_fs);

    int num = config_doc.MemberEnd() - config_doc.MemberBegin();
    std::vector<int> record_batch_size;
    int idx = 0;
    for (auto& m : config_doc.GetObject()){
        string name = to_string(idx);
        rapidjson::Value& array = m.value;
        size_t len = array.Size();
        record_batch_size.push_back(0);
        for(size_t i = 0; i < len; i++)
        {
            record_batch_size[idx] += array[i].GetBool();
        }
        idx += 1;
    }
    return record_batch_size;
}

std::vector<float> Profiler::bert_execute(const bool separate_or_not, const size_t& num_test,
                 const std::vector<int> record_batch_size, const int copy_method, const bool overload, std::string model_name)
{
    float elapsed_time = 0;
    std::vector<float> metrics;
    if (separate_or_not) {

        size_t sequence_length = 7;

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

        samplesCommon::BertBufferManager buffer_s1(mEngine_list[3], batch_size_s1_);
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

        int next_batch_size = record_batch_size[std::rand()%(record_batch_size.size())];
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
        size_t sequence_length = 7;
        if (begin_point_ == 0) {
            input_dims_s1.d[0] = batch_size_s1_;
            input_dims_s1.d[1] = sequence_length;
            mContext_s0->setBindingDimensions(0, input_dims_s1);
            mContext_s0->setBindingDimensions(1, input_dims_s1);
            mContext_s0->setBindingDimensions(2, input_dims_s1);
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

std::vector<float> Profiler::execute_2stage(const bool separate_or_not, const size_t& num_test,
                 const std::vector<int> record_batch_size, const int copy_method, const bool overload, std::string model_name)
{
    float elapsed_time = 0;
    std::vector<float> metrics;

    int engine_idx_1 = 3;
    input_dims[engine_idx_1].d[0] = batch_size_s1_;
    std::cout << input_dims[engine_idx_1] <<std::endl;
    mContext_list[engine_idx_1]->setBindingDimensions(0, input_dims[engine_idx_1]);
    samplesCommon::BufferManager buffer_s1(mEngine_list[engine_idx_1], batch_size_s1_);
    buffer_s1.copyInputToDeviceAsync(stream_1);
    auto status_s1 = mContext_list[engine_idx_1]->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
    if (!status_s1) {
        std::cout << "Error when inferring S1 model" << std::endl;
    }

    std::shared_ptr<samplesCommon::ManagedBuffer> exitPtr = buffer_s1.getImmediateBuffer(2);
        
    float* exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
    int *copy_list;
    int size = (int) batch_size_s1_*sizeof(int);
    cudaMalloc(&copy_list, size);
    max_reduction_r(exitPtr_device, copy_list, stream_1);

    int next_batch_size = batch_size_s1_;
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

    int engine_idx_2 = (next_batch_size-1)/(batch_size_s1_/4)+4;
    input_dims[engine_idx_2].d[0] = next_batch_size;
    std::cout << input_dims[engine_idx_2] << std::endl;
    mContext_list[engine_idx_2]->setBindingDimensions(0, input_dims[engine_idx_2]);

    std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = buffer_s1.getImmediateBuffer(1);
    samplesCommon::BufferManager buffer_s2(mEngine_list[engine_idx_2], batch_size_s1_,
                            srcPtr, fake_copy_list, next_batch_size, copy_method);

    auto status_s2 = mContext_list[engine_idx_2]->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
    if (!status_s2) {
        std::cout << "Error when inferring S2 model" << std::endl;
    }

    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaEventRecord(infer_start, stream_1));
    for (int i = 0; i < batch_num_; i++){
        // buffer_s1.copyInputToDeviceAsync(stream_2);
        // CUDACHECK(cudaDeviceSynchronize());

        engine_idx_1 = 3;
        mContext_list[engine_idx_1]->setBindingDimensions(0, input_dims[engine_idx_1]);
        status_s1 = mContext_list[engine_idx_1]->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
        if (!status_s1) {
            std::cout << "Error when inferring S1 model" << std::endl;
        }

        exitPtr = buffer_s1.getImmediateBuffer(2);
        exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
        max_reduction_r(exitPtr_device, copy_list, stream_1);

        next_batch_size = record_batch_size[std::rand()%(record_batch_size.size())];
        // next_batch_size = 128;

        if (next_batch_size == 0) {
            CUDACHECK(cudaDeviceSynchronize());
            continue;
        }

        engine_idx_2 = (next_batch_size-1)/(batch_size_s1_/4)+4;
        std::shared_ptr<samplesCommon::ManagedBuffer> manBuf_ptr = buffer_s1.getImmediateBuffer(1);
        std::shared_ptr<samplesCommon::ManagedBuffer> new_manBuf = buffer_s2.getImmediateBuffer(0);
        float* dstPtr_ = static_cast<float*>(new_manBuf->deviceBuffer.data());
        float* srcPtr_ = static_cast<float*>(manBuf_ptr->deviceBuffer.data());
        auto dims = mEngine_list[engine_idx_2]->getBindingDimensions(0);
        dims.d[0] = 1;
        size_t singleVol = samplesCommon::volume(dims);

        buffercopy(dstPtr_, srcPtr_, singleVol*next_batch_size, fake_copy_list, singleVol, stream_1);

        input_dims[engine_idx_2].d[0] = next_batch_size;
        mContext_list[engine_idx_2]->setBindingDimensions(0, input_dims[engine_idx_2]);
        status_s2 = mContext_list[engine_idx_2]->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_0, nullptr);
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
    return metrics;
}


std::vector<float> Profiler::execute_multi_stage(const bool separate_or_not, const size_t& num_test,
                 const std::vector<std::vector<int>> record_batch_size, const int copy_method, const bool overload, std::string model_name)
{
    float elapsed_time = 0;
    std::vector<float> metrics;

    // Generate fake copy list on the CPU side and copy to the GPU
    int next_batch_size = batch_size_s1_;
    int *fake_copy_list;
    cudaMalloc(&fake_copy_list, (int) next_batch_size*sizeof(int));
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

    /* Engine1 Initilaize */
    int engine_idx_1 = 3;
    input_dims[engine_idx_1].d[0] = batch_size_s1_;
    std::cout << "Initializing engine1. Input dimension: " << input_dims[engine_idx_1] <<std::endl;
    mContext_list[engine_idx_1]->setBindingDimensions(0, input_dims[engine_idx_1]);
    samplesCommon::BufferManager buffer_s1(mEngine_list[engine_idx_1], batch_size_s1_);
    // Initialize buffer_s2
    buffer_s1.copyInputToDeviceAsync(stream_1);
    auto status_s1 = mContext_list[engine_idx_1]->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
    if (!status_s1) {
        std::cout << "Error when inferring S1 model" << std::endl;
    }
    // Max_reduction
    std::shared_ptr<samplesCommon::ManagedBuffer> exitPtr = buffer_s1.getImmediateBuffer(2);
    float* exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
    int *copy_list;
    int size = (int) batch_size_s1_*sizeof(int);
    cudaMalloc(&copy_list, size);
    max_reduction_r(exitPtr_device, copy_list, stream_1);

    /* Engine2 Initilaize */
    int engine_idx_2 = (next_batch_size-1)/(batch_size_s1_/4)+4;
    input_dims[engine_idx_2].d[0] = next_batch_size;
    std::cout << "Initializing engine2. Input dimension: " << input_dims[engine_idx_2] << std::endl;
    mContext_list[engine_idx_2]->setBindingDimensions(0, input_dims[engine_idx_2]);
    // Initialize buffer_s2
    std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = buffer_s1.getImmediateBuffer(1);
    samplesCommon::BufferManager buffer_s2(mEngine_list[engine_idx_2], batch_size_s1_,
                            srcPtr, fake_copy_list, next_batch_size, copy_method);

    auto status_s2 = mContext_list[engine_idx_2]->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_1, nullptr);
    if (!status_s2) {
        std::cout << "Error when inferring S2 model" << std::endl;
    }
    // Max_reduction
    exitPtr = buffer_s2.getImmediateBuffer(2);
    exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
    max_reduction_r(exitPtr_device, copy_list, stream_1);


    /* Engine3 Initialize */
    int engine_idx_3 = (next_batch_size-1)/(batch_size_s1_/4)+8;
    input_dims[engine_idx_3].d[0] = next_batch_size;
    std::cout << "Initializing engine3. Input dimension: " << input_dims[engine_idx_2] << std::endl;
    mContext_list[engine_idx_3]->setBindingDimensions(0, input_dims[engine_idx_3]);
    srcPtr = buffer_s2.getImmediateBuffer(1);
    samplesCommon::BufferManager buffer_s3(mEngine_list[engine_idx_3], batch_size_s1_,
                            srcPtr, fake_copy_list, next_batch_size, copy_method);
    auto status_s3 = mContext_list[engine_idx_3]->enqueueV2(buffer_s3.getDeviceBindings().data(), stream_1, nullptr);
    if (!status_s3) {
        std::cout << "Error when inferring S3 model" << std::endl;
    }

    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(infer_start, stream_1));
    std::cout << "Begin recording..." << std::endl;
    for (int i = 0; i < batch_num_; i++){
        // buffer_s1.copyInputToDeviceAsync(stream_2);
        // CUDACHECK(cudaDeviceSynchronize());

        engine_idx_1 = 3;
        mContext_list[engine_idx_1]->setBindingDimensions(0, input_dims[engine_idx_1]);
        status_s1 = mContext_list[engine_idx_1]->enqueueV2(buffer_s1.getDeviceBindings().data(), stream_1, nullptr);
        if (!status_s1) {
            std::cout << "Error when inferring S1 model" << std::endl;
        }

        exitPtr = buffer_s1.getImmediateBuffer(2);
        exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
        max_reduction_r(exitPtr_device, copy_list, stream_1);

        next_batch_size = record_batch_size[0][std::rand()%(record_batch_size[0].size())];
        if (next_batch_size == 0) {
            CUDACHECK(cudaDeviceSynchronize());
            continue;
        }

        engine_idx_2 = (next_batch_size-1)/(batch_size_s1_/4)+4;
        std::shared_ptr<samplesCommon::ManagedBuffer> manBuf_ptr = buffer_s1.getImmediateBuffer(1);
        std::shared_ptr<samplesCommon::ManagedBuffer> new_manBuf = buffer_s2.getImmediateBuffer(0);
        float* dstPtr_ = static_cast<float*>(new_manBuf->deviceBuffer.data());
        float* srcPtr_ = static_cast<float*>(manBuf_ptr->deviceBuffer.data());
        auto dims = mEngine_list[engine_idx_2]->getBindingDimensions(0);
        dims.d[0] = 1;
        size_t singleVol = samplesCommon::volume(dims);

        buffercopy(dstPtr_, srcPtr_, singleVol*next_batch_size, fake_copy_list, singleVol, stream_1);

        input_dims[engine_idx_2].d[0] = next_batch_size;
        mContext_list[engine_idx_2]->setBindingDimensions(0, input_dims[engine_idx_2]);
        status_s2 = mContext_list[engine_idx_2]->enqueueV2(buffer_s2.getDeviceBindings().data(), stream_1, nullptr);
        if (!status_s2) {
            std::cout << "Error when inferring S2 model" << std::endl;
        }

        exitPtr = buffer_s2.getImmediateBuffer(2);
        exitPtr_device = static_cast<float*>(exitPtr->deviceBuffer.data());
        max_reduction_r(exitPtr_device, copy_list, stream_1);

        next_batch_size = record_batch_size[1][std::rand()%(record_batch_size[1].size())];
        if (next_batch_size == 0) {
            CUDACHECK(cudaDeviceSynchronize());
            continue;
        }

        engine_idx_3 = (next_batch_size-1)/(batch_size_s1_/4)+8;
        manBuf_ptr = buffer_s2.getImmediateBuffer(1);
        new_manBuf = buffer_s3.getImmediateBuffer(0);
        dstPtr_ = static_cast<float*>(new_manBuf->deviceBuffer.data());
        srcPtr_ = static_cast<float*>(manBuf_ptr->deviceBuffer.data());

        buffercopy(dstPtr_, srcPtr_, singleVol*next_batch_size, fake_copy_list, singleVol, stream_1);

        input_dims[engine_idx_3].d[0] = next_batch_size;
        mContext_list[engine_idx_3]->setBindingDimensions(0, input_dims[engine_idx_3]);
        status_s3 = mContext_list[engine_idx_3]->enqueueV2(buffer_s3.getDeviceBindings().data(), stream_1, nullptr);
        if (!status_s3) {
            std::cout << "Error when inferring S3 model" << std::endl;
        }

        CUDACHECK(cudaDeviceSynchronize());
    }

    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(s2_end, stream_0));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventElapsedTime(&elapsed_time, infer_start, s2_end));
    metrics.push_back(elapsed_time);
    return metrics;
}




bool model_generation(std::string model_name, const int start_point, const int end_point)
{
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/slzhang/projects/ETBA/Inference/src/run_engine')");
	PyObject* pModule = PyImport_ImportModule("model_export_v2");
	if( pModule == NULL ){
		cout <<"module not found" << endl;
		return 1;
	}
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
    // int nGpuId = 2;
    // cudaSetDevice(nGpuId);
    // int leastPriority;
    // int greatestPriority;
    // cudaDeviceGetStreamPriorityRange (&leastPriority, &greatestPriority );
    // std::cout << leastPriority << "  " << greatestPriority << std::endl;

    std::string model_name = argv[1];
    std::cout << "Profiling model: " << model_name + "!" << std::endl;
    std::string config_path = "/home/slzhang/projects/ETBA/Inference/src/run_engine/profiler_config.json";
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
    Py_Initialize();
    int split_point = config_doc["split_point"].GetUint();
    // bool model_generated = model_generation(model_name, config_doc["begin_point"].GetUint(), split_point);
    // if(!model_generated){
    //     std::cout<<"failed to export models"<<endl;
    //     return -1;
    // }
    
    std::vector<float> elapsed_time;
    int infer_batch_size_s1 = config_doc["bs_s1"].GetUint();
    int infer_batch_size_s2 = config_doc["bs_s2"].GetUint();
    // size_t infer_batch_size_s2 = config_doc["bs_s2"].GetUint() * batch_scale / 4;
    Profiler inst = Profiler(infer_batch_size_s1, infer_batch_size_s2, 
                            config_doc["bs_num"].GetUint(), config_doc["begin_point"].GetUint(),
                            nvinfer1::ILogger::Severity::kERROR);

    std::string movon_dict_path_1 = "/home/slzhang/projects/ETBA/Train/moveon_dict/resnet_exit_e9_b128.json";
    std::string movon_dict_path_2 = "/home/slzhang/projects/ETBA/Train/moveon_dict/resnet_exit_e22_l9_b128.json";
    std::vector<int> record_batch_size = generate_copy_list(movon_dict_path_1);
    std::vector<int> record_batch_size_2 = generate_copy_list(movon_dict_path_2);
    std::vector<std::vector<int>> multi_record_batch_size;
    multi_record_batch_size.push_back(record_batch_size);
    multi_record_batch_size.push_back(record_batch_size_2);
    std::vector<std::string> model_name_list;
    model_name_list.push_back("resnet_stage1");
    model_name_list.push_back("resnet_stage2");
    model_name_list.push_back("resnet_stage3");
    std::cout << "Building engines ..." << std::endl;
    inst.build(model_name_list, 128, 4);
    std::cout << "Building finished!" << std::endl;

    std::vector<float> metrics;
    if (model_name == "bert"){
        metrics = inst.bert_execute(config_doc["seperate_or_not"].GetBool(),
                                    config_doc["test_iter"].GetUint(), record_batch_size, 
                                    config_doc["copy_method"].GetUint(), false, model_name);
    }
    else {
        // metrics = inst.execute(config_doc["seperate_or_not"].GetBool(),
        //                         config_doc["test_iter"].GetUint(), record_batch_size, 
        //                         config_doc["copy_method"].GetUint(), false, model_name);
        metrics = inst.execute_multi_stage(config_doc["seperate_or_not"].GetBool(),
                                config_doc["test_iter"].GetUint(), multi_record_batch_size, 
                                config_doc["copy_method"].GetUint(), false, model_name);                  
    }

    std::cout << "Batch size: " << infer_batch_size_s2 << "/" << infer_batch_size_s1 << "  Elapsed time: " << metrics[0]/inst.batch_num_ << std::endl;
    elapsed_time.push_back(metrics[0]/inst.batch_num_);
    
    outFile.open("/home/slzhang/projects/ETBA/Inference/src/run_engine/config_" + model_name + "_" +
                    to_string(config_doc["bs_s1"].GetUint()) + ".csv", ios::app);
    outFile << 0 << ',' << split_point << ',';

    for (int i = 0; i < elapsed_time.size(); ++i){
        outFile << elapsed_time.at(i) << ',';
        if (i == elapsed_time.size() - 1) {outFile << endl;}
    }

    outFile.close();
    Py_Finalize();
    return 0;
}
/*
A sample for vgg16 inference
*/

#include <cuda_runtime.h>
#include "sample_vgg16.h"

Profiler::Profiler(
const ProfilerConfig& profiler_config, const size_t& min_batch_size,
const size_t& opt_batch_size, const size_t& max_batch_size, const int batch_num, 
const Severity severity)
: min_batch_size_(min_batch_size), opt_batch_size_(opt_batch_size),
  max_batch_size_(max_batch_size), batch_num_(batch_num)
{
    cudaStreamCreate(&(stream_[0]));
    cudaStreamCreate(&(stream_[1]));
    CUDACHECK(cudaEventCreate(&start_));
    CUDACHECK(cudaEventCreate(&stop_));

    profiler_config_ = profiler_config;
    size_t sub_model_cnt = profiler_config_.getSegNum();
    size_t ee_model_cnt = profiler_config_.geteeNum();
        
    for (size_t i = 0; i < sub_model_cnt; i++) {
        CUDACHECK(cudaEventCreate(&ms_stop_[i]));
    }
    for (size_t i = 0; i < ee_model_cnt; i++) {
        CUDACHECK(cudaEventCreate(&ee_stop_[i]));
    }
        
    sample::gLogger.setReportableSeverity(severity);
}

bool Profiler::build()
{
    size_t sub_model_cnt = profiler_config_.getSegNum();
    size_t ee_model_cnt = profiler_config_.geteeNum();
    sub_engines_.clear();
    sub_contexts_.clear();
    ee_engines_.clear();
    ee_contexts_.clear();

    sub_input_dims_.resize(sub_model_cnt);
    sub_input_tensor_names_.resize(sub_model_cnt);
    sub_output_dims_.resize(sub_model_cnt);
    sub_output_tensor_names_.resize(sub_model_cnt);

    ee_input_dims_.resize(ee_model_cnt);
    ee_input_tensor_names_.resize(ee_model_cnt);
    ee_output_dims_.resize(ee_model_cnt);
    ee_output_tensor_names_.resize(ee_model_cnt);
    ee_indicator.resize(ee_model_cnt);
    accuracy.resize(batch_num_);

    for (size_t i = 0; i < sub_model_cnt; i++) {
        // sub_buffer_manager_.emplace_back(std::move(BufferManager()));
        auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder) {
            std::cout << "Failed to create " << i << "-th sub builder";
            return false;
        }
        const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network) {
            std::cout << "Failed to create " << i << "-th sub network";
            return false;
        }
        auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser) {
            std::cout << "Failed to create " << i << "-th sub parser";
            return false;
        }
        auto constructed = constructSubNet(builder, network, config, parser, i);
        if (!constructed) {
            std::cout << "Failed to construct " << i << "-th sub network";
            return false;
        }

        auto tmp_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!tmp_engine) {
            std::cout << "Failed to create " << i << "-th sub engine";
            return false;
        }

        sub_engines_.emplace_back(tmp_engine);
        std::cout << tmp_engine->getBindingName(0) << std::endl;
        std::cout << sub_engines_[i]->getNbBindings() << std::endl;
        sub_contexts_.emplace_back(std::shared_ptr<nvinfer1::IExecutionContext>(
            tmp_engine->createExecutionContext(), samplesCommon::InferDeleter()));
    }

    for (size_t i = 0; i < ee_model_cnt; i++) {
        auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder) {
            std::cout << "Failed to create " << i << "-th ee builder";
            return false;
        }
        const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network) {
            std::cout << "Failed to create " << i << "-th ee network";
            return false;
        }
        auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser) {
            std::cout << "Failed to create " << i << "-th ee parser";
            return false;
        }
        auto constructed = constructeeNet(builder, network, config, parser, i);
        if (!constructed) {
            std::cout << "Failed to construct " << i << "-th ee network";
            return false;
        }

        auto tmp_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!tmp_engine) {
            std::cout << "Failed to create " << i << "-th ee engine";
            return false;
        }

        ee_engines_.emplace_back(tmp_engine);
        std::cout << tmp_engine->getBindingName(0) << std::endl;
        //std::cout << sub_engines_[i]->getBindingIndex(sub_input_tensor_names_[i][0]) << std::endl;
        std::cout << ee_engines_[i]->getNbBindings() << std::endl;
        //ee_contexts_[i] = std::shared_ptr<nvinfer1::IExecutionContext>(tmp_engine->createExecutionContext(), samplesCommon::InferDeleter());
        std::cout << "The " << i << "-th ee engine built successfully" << std::endl;
        ee_contexts_.emplace_back(std::shared_ptr<nvinfer1::IExecutionContext>(
            tmp_engine->createExecutionContext(), samplesCommon::InferDeleter()));
    }
    readData();
    std::cout << "Read data successfully" << std::endl;
    return true;
}

bool Profiler::constructSubNet(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("src/sample_vgg16/vgg_model/cifar10/onnx_main_arc3");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile("main_arch_"+to_string(model_index)+".onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    sub_input_dims_[model_index] = network->getInput(0)->getDimensions();
    sub_input_tensor_names_[model_index] = network->getInput(0)->getName();
    //std:cout << network->getInput(0)->getName() << std::endl;

    sub_output_dims_[model_index] = network->getOutput(0)->getDimensions();
    sub_output_tensor_names_[model_index] = network->getOutput(0)->getName();

    nvinfer1::Dims min_dims = sub_input_dims_[model_index];
    min_dims.d[0] = 1;
    nvinfer1::Dims opt_dims = sub_input_dims_[model_index];
    opt_dims.d[0] = 1;
    nvinfer1::Dims max_dims = sub_input_dims_[model_index];
    max_dims.d[0] = max_batch_size_;

    profile->setDimensions(sub_input_tensor_names_[model_index].c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(sub_input_tensor_names_[model_index].c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(sub_input_tensor_names_[model_index].c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(3_GiB);
    if (profiler_config_.fp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    return true;
}

bool Profiler::constructeeNet(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index)
{
    auto profile = builder->createOptimizationProfile();
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("src/sample_vgg16/vgg_model/cifar10/onnx_IC3");
    //data_dir.push_back("samples/VGG16/");
    auto parsed = parser->parseFromFile(locateFile("IC_"+to_string(model_index)+".onnx", params.dataDirs).c_str(),
    static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    ee_input_dims_[model_index] = network->getInput(0)->getDimensions();
    ee_input_tensor_names_[model_index] = network->getInput(0)->getName();

    ee_output_dims_[model_index] = network->getOutput(0)->getDimensions();
    ee_output_tensor_names_[model_index] = network->getOutput(0)->getName();

    nvinfer1::Dims min_dims = ee_input_dims_[model_index];
    min_dims.d[0] = 1;
    nvinfer1::Dims opt_dims = ee_input_dims_[model_index];
    opt_dims.d[0] = 1;
    nvinfer1::Dims max_dims = ee_input_dims_[model_index];
    max_dims.d[0] = max_batch_size_;

    profile->setDimensions(ee_input_tensor_names_[model_index].c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(ee_input_tensor_names_[model_index].c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(ee_input_tensor_names_[model_index].c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(3_GiB);
    if (profiler_config_.fp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    return true;
}

bool Profiler::infer(const size_t& num_test, const size_t& batch_size, const int batch_idx)
{
    // Read the input data into the managed stage1 buffers
    // assert(mParams.inputTensorNames.size() == 1);
    size_t sub_model_cnt = profiler_config_.getSegNum();
    size_t ee_model_cnt = profiler_config_.geteeNum();
    ee_batch_size.resize(ee_model_cnt);
    sub_batch_size.resize(sub_model_cnt);
    size_t init_batch_size = max_batch_size_;
    sub_batch_size[0] = init_batch_size;
    ee_batch_size[0] = init_batch_size;

    subToEE[1] = 0;
    subToEE[3] = 1;
    subToEE[5] = 2;
    subToEE[6] = 3;
    subToEE[8] = 4;
    subToEE[9] = 5;
    
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Infering......" << std::endl;
    //CUDACHECK(cudaEventRecord(start_, stream_[0]));

    for (size_t i = 0; i < sub_model_cnt; i++){
        std::cout << "Stage index: " << i << std::endl;

        // Start stage: stage_type = 0
        if (stage_type[i] == 0){
            //CUDACHECK(cudaEventRecord(ms_stop_[i], stream_[0]));
            
            CUDACHECK(cudaDeviceSynchronize());
            sub_input_dims_[i].d[0] = sub_batch_size[i];
            sub_contexts_[i]->setBindingDimensions(0, sub_input_dims_[i]);
            samplesCommon::BufferManager tmp_buffer(sub_engines_[i], sub_batch_size[i]);
            sub_buffer_manager_.emplace_back(std::move(tmp_buffer));
            sub_batch_size[i+1] = sub_batch_size[i];

            if (!processInput(sub_buffer_manager_[i], 0))
            {
                return false;
            }
            // Memcpy from host input buffers to device input buffers
            sub_buffer_manager_[i].copyInputToDevice();

            auto status1 = sub_contexts_[i]->enqueueV2(sub_buffer_manager_[i].getDeviceBindings().data(), stream_[0], nullptr);
            if (!status1) {
                std::cout << "Error when inference " << i << "-th sub model" << std::endl;
                return false;
            }
            CUDACHECK(cudaEventRecord(ms_stop_[i], stream_[0]));
            CUDACHECK(cudaEventRecord(start_, stream_[0]));
        }
        // Parallel-with-IC stage: stage_type = 1
        else if (stage_type[i] == 1){
            CUDACHECK(cudaStreamWaitEvent(stream_[0], start_));
            CUDACHECK(cudaStreamWaitEvent(stream_[0], ms_stop_[i-1]));
            CUDACHECK(cudaStreamWaitEvent(stream_[1], ms_stop_[i-1]));
            size_t map_i = subToEE[i];

            // construct the sub buffer
            sub_input_dims_[i].d[0] = sub_batch_size[i];
            sub_contexts_[i]->setBindingDimensions(0, sub_input_dims_[i]);
            std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = sub_buffer_manager_[i-1].getOutputBuffer();
            if (stage_type[i-1] == 1){
                // Two sources
                samplesCommon::BufferManager tmp_buffer1(sub_engines_[i], sub_batch_size[i], srcPtr, &ee_indicator[subToEE[i-1]]);
                sub_buffer_manager_.emplace_back(std::move(tmp_buffer1));
            }
            else {
                // One source
                samplesCommon::BufferManager tmp_buffer1(sub_engines_[i], sub_batch_size[i], srcPtr);
                sub_buffer_manager_.emplace_back(std::move(tmp_buffer1));
            }
            //std::cout << "Buffer management of the " << i << "-th sub buffer done. " << std::endl;

            auto status3 = sub_contexts_[i]->enqueueV2(sub_buffer_manager_[i].getDeviceBindings().data(), stream_[0], nullptr);
            if (!status3) {
                std::cout << "Error when inference " << i << "-th sub model" << std::endl;
                return false;
            }
            CUDACHECK(cudaEventRecord(ms_stop_[i], stream_[0]));

            //construct the ee buffer (the same buffer as the input of the parallel main arch buffer)
            ee_batch_size[map_i] = sub_batch_size[i];
            std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr_parallel = sub_buffer_manager_[i].getInputBuffer();
            samplesCommon::BufferManager tmp_buffer2(ee_engines_[map_i], ee_batch_size[map_i], srcPtr_parallel);
            ee_buffer_manager_.emplace_back(std::move(tmp_buffer2));

            ee_input_dims_[map_i].d[0] = ee_batch_size[map_i];
            ee_contexts_[map_i]->setBindingDimensions(0, ee_input_dims_[map_i]);
            auto status2 = ee_contexts_[map_i]->enqueueV2(ee_buffer_manager_[map_i].getDeviceBindings().data(), stream_[1], nullptr);
            if (!status2) {
                std::cout << "Error when inference " << i << "-th ee model" << std::endl;
                return false;
            }

            auto controlled = controller(i, map_i);
            if (!controlled) {
                std::cout << "Controller " << i << " broken" << std::endl;
                return false;
            }       

            CUDACHECK(cudaEventRecord(ee_stop_[subToEE[i]], stream_[1]));
        }
        // Single Main Arch Stage: stage_type = 2
        else if (stage_type[i] == 2){
            CUDACHECK(cudaStreamWaitEvent(stream_[0], ee_stop_[i/2-1]));
            CUDACHECK(cudaStreamWaitEvent(stream_[0], ms_stop_[i-1]));

            sub_input_dims_[i].d[0] = sub_batch_size[i];
            sub_contexts_[i]->setBindingDimensions(0, sub_input_dims_[i]);

            std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = sub_buffer_manager_[i-1].getOutputBuffer();
            //std::shared_ptr<samplesCommon::ManagedBuffer> eeResultPtr = ee_buffer_manager_[i/2-1].getOutputBuffer();
            samplesCommon::BufferManager tmp_buffer(sub_engines_[i], sub_batch_size[i], srcPtr, &ee_indicator[subToEE[i-1]]);
            sub_buffer_manager_.emplace_back(std::move(tmp_buffer));
            //std::cout << "Buffer management of the " << i << "-th sub buffer done. " << std::endl;

            sub_batch_size[i+1] = sub_batch_size[i];

            auto status4 = sub_contexts_[i]->enqueueV2(sub_buffer_manager_[i].getDeviceBindings().data(), stream_[0], nullptr);
            if (!status4) {
                std::cout << "Error when inference " << i << "-th sub model" << std::endl;
                return false;
            }
            CUDACHECK(cudaEventRecord(ms_stop_[i], stream_[0]));
        }
        // End Stage: stage_type = 3
        else if (stage_type[i] == 3){
            CUDACHECK(cudaStreamWaitEvent(stream_[0], ms_stop_[i-1]));
            sub_input_dims_[i].d[0] = sub_batch_size[i];
            sub_contexts_[i]->setBindingDimensions(0, sub_input_dims_[i]);
            std::shared_ptr<samplesCommon::ManagedBuffer> srcPtr = sub_buffer_manager_[i-1].getOutputBuffer();
            samplesCommon::BufferManager tmp_buffer(sub_engines_[i], sub_batch_size[i], srcPtr);
            sub_buffer_manager_.emplace_back(std::move(tmp_buffer));
            //std::cout << "Buffer management of the " << i << "-th sub buffer done. " << std::endl;

            auto status6 = sub_contexts_[i]->enqueueV2(sub_buffer_manager_[i].getDeviceBindings().data(), stream_[0], nullptr);
            if (!status6) {
                std::cout << "Error when inference " << i << "-th sub model" << std::endl;
                return false;
            }
            sub_buffer_manager_[i].copyOutputToHost();
            accuracy[batch_idx] = verifyOutput(sub_buffer_manager_[i], 0);

            std::cout << "Inference finished!" << std::endl;
            CUDACHECK(cudaEventRecord(ms_stop_[i], stream_[0]));
            CUDACHECK(cudaEventRecord(stop_, stream_[0]));
            CUDACHECK(cudaEventSynchronize(stop_));
        }
        std::cout << "=========================================================" << std::endl;
    }

    std::vector<float> milli_sec;
    milli_sec.resize(11);

    for (int i=0; i < 10; i++) {
        CUDACHECK(cudaEventElapsedTime(&milli_sec[i], ms_stop_[i], ms_stop_[i+1]));
        std::cout << "Elapsed time " << i+1 << ": " << milli_sec[i] << std::endl; 
    }
        
    float total_elapsed_time = 0;
    CUDACHECK(cudaEventElapsedTime(&total_elapsed_time, ms_stop_[0], stop_));
    std::cout << "Total elapsed time: " << total_elapsed_time << std::endl;

    return true;
}

std::vector<void*> Profiler::getDeviceBindings(const size_t& model_index)
{
    return sub_buffer_manager_[model_index].getDeviceBindings();
}

bool Profiler::controller(const int stage_idx, const int ee_idx)
{
    ee_indicator[ee_idx].resize(ee_batch_size[ee_idx]);
    std::shared_ptr<samplesCommon::ManagedBuffer> eeResultPtr = ee_buffer_manager_[ee_idx].getOutputBuffer();
    const cudaMemcpyKind memcpyType = cudaMemcpyDeviceToHost;
    cudaMemcpy(eeResultPtr->hostBuffer.data(), eeResultPtr->deviceBuffer.data(),
                eeResultPtr->hostBuffer.nbBytes(), memcpyType);
    float *res = static_cast<float*>(eeResultPtr->hostBuffer.data());
    std::cout << "Indicator length: " << ee_batch_size[ee_idx] << std::endl;
    for (size_t j = 0; j < ee_batch_size[ee_idx]; j++){
        int maxposition = std::max_element(res+10*j, res+10*j + 10) - (res+10*j);
        // std::cout << "max value: " << *(res+10*j + maxposition) << std::endl;
        ee_indicator[ee_idx][j] = (*(res+10*j + maxposition) > profiler_config_.ee_thresholds_[subToEE[stage_idx]]) ? 1 : 0;
    }
    sub_batch_size[stage_idx+1] = std::accumulate(ee_indicator[ee_idx].begin(), ee_indicator[ee_idx].end(), 0);
    //std::cout << "Batch size of next stage: " << sub_batch_size[stage_idx+1] << std::endl;

    return true;
}

float Profiler::verifyOutput(const samplesCommon::BufferManager& buffer, const int batch_idx = 0)
{
    const int inputC = 3;
    const int inputH = 32;
    const int inputW = 32;
    //const int batchSize = mParams.batchSize;
    const int volImg = inputC * inputH * inputW;
    const int imageSize = volImg + 1;
    int i = 0;
    float* output = static_cast<float*>(buffer.getHostBuffer(sub_output_tensor_names_[10]));
    int maxposition{0};
    int count{0};
    //for (size_t i = 0; i < 10; i++) {
    //    std::cout << *(output + i) << std::endl;
    //}
    for (size_t i = 0; i < sub_batch_size.back(); i++) {
        maxposition = std::max_element(output+10*i, output+10*i + 10) - (output+10*i);
        //std::cout << "maxposition: " << maxposition << " correctposition: " << int(cifarbinary[(i+32) * imageSize]) << endl;
        if (maxposition == int(cifarbinary[(i + max_batch_size_ * batch_idx) * imageSize])) {
            ++count;
        }
    }
    //std::cout << "The number of correct samples is: " << count << endl;
    float accuracy = float(count) / float(sub_batch_size.back());
    std::cout << "The accuracy of the TRT Engine on " << sub_batch_size.back() << " data is: " << accuracy << endl;
    return accuracy;
}


bool Profiler::readData()
{
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("data/cifar10");
    // 5 batchbinary files
    for (int index = 0; index < 5; ++index) {
        // Read cifar10 original binary file
        readBinaryFile(locateFile("data_batch_" + std::to_string(index + 1) + ".bin", params.dataDirs), cifarbinary);
    }
    return true;
}

bool Profiler::processInput(const samplesCommon::BufferManager& buffer, const int batch_idx = 0)
{
    std::cout << "Pre processing begin." << std::endl;
    const int inputC = 3;
    const int inputH = 32;
    const int inputW = 32;
    //const int batchSize = mParams.batchSize;
    const int volImg = inputC * inputH * inputW;
    const int imageSize = volImg + 1;
    const int outputSize = 10;
    float* hostDataBuffer = static_cast<float*>(buffer.getHostBuffer(sub_input_tensor_names_[0]));
      
        for (int i = 0; i < max_batch_size_; ++i) {
            for (int j = 0; j < 32 * 32 * 3; ++j) {
                //RGB format
                if (j < 32*32) {
                    hostDataBuffer[i*volImg+j] = (float(cifarbinary[(i + max_batch_size_ * batch_idx) * imageSize + j])/255.0-0.485)/0.229;
                }
                else if (j < 32*32*2) {
                    hostDataBuffer[i*volImg+j] = (float(cifarbinary[(i + max_batch_size_ * batch_idx) * imageSize + j])/255.0-0.456)/0.224;
                }
                else {
                    hostDataBuffer[i*volImg+j] = (float(cifarbinary[(i + max_batch_size_ * batch_idx) * imageSize + j])/255.0-0.406)/0.225;
                }
            }
         }
    std::cout << "Pre processing finished." << std::endl;
    return true;
}

ProfilerConfig getInstConfig(const rapidjson::Document& config_doc)
{
    assert(config_doc["seg_files"].IsArray());
    const rapidjson::Value& model_files = config_doc["seg_files"];
    const rapidjson::Value& thresholds = config_doc["thresholds"];
    assert(model_files.Size() == config_doc["seg_num"].GetUint());
    ProfilerConfig profiler_config{config_doc["seg_num"].GetUint(),
        config_doc["ee_num"].GetUint(), config_doc["dir"].GetString(),
        nvinfer1::DataType::kFLOAT, config_doc["fp16"].GetBool()};
    for (rapidjson::SizeType i = 0; i < model_files.Size(); i++) {
        profiler_config.setSegFileName(i, model_files[i].GetString());
    }
    for (rapidjson::SizeType i = 0; i < thresholds.Size(); i++) {
        profiler_config.ee_thresholds_[i] = float(thresholds[i].GetDouble());
    }
    return profiler_config;
}

int main(int argc, char** argv)
{
    //std::string config_path = argv[1];
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::string config_path = "../profiler_config.json";
    //std::string config_path = "profiler_config.json";
    std::cout << config_path << std::endl;
    FILE* config_fp = fopen(config_path.c_str(), "r");
    char read_buffer[65536];

    rapidjson::FileReadStream config_fs(
        config_fp, read_buffer, sizeof(read_buffer));
    rapidjson::Document config_doc;

    config_doc.ParseStream(config_fs);
    ProfilerConfig profiler_config = getInstConfig(config_doc);

    Profiler inst = Profiler(
        profiler_config, config_doc["min_bs"].GetUint(),
        config_doc["opt_bs"].GetUint(), config_doc["max_bs"].GetUint(), config_doc["bs_num"].GetUint(),
        nvinfer1::ILogger::Severity::kINFO);

    inst.build();
    for (int batch_idx = 0; batch_idx < inst.batch_num_; batch_idx++) {
        inst.infer(config_doc["test_iter"].GetUint(), config_doc["cur_bs"].GetUint(), batch_idx);
        std::cout << inst.accuracy[batch_idx] << std::endl;
    }
    //inst.exportTrtModel(profiler_config.getDataDir());
    return 0;
}
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers_ori.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_cifar10_vgg16bn";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
    std::vector<uint8_t> cifarbinary;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    int max_batch_size_;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers,const int batch_idx);
    bool readData();
    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers, const int batch_idx);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    std::cout<<"build begin"<<endl;
    max_batch_size_ = 256;
    std::cout<<"batch size is: "<<max_batch_size_<<endl;

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    std::cout<<"construct begin"<<endl;
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }
    std::cout<<"construct end"<<endl;

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    readData();

    std::cout<<"build finish"<<endl;
    return true;
}

bool SampleOnnxMNIST::readData()
{
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.emplace_back("/home/slzhang/projects/ETBA/Inference/data/cifar10");
    // 5 batchbinary files
    for (int index = 0; index < 5; ++index) {
        // Read cifar10 original binary file
        readBinaryFile(locateFile("data_batch_" + std::to_string(index + 1) + ".bin", params.dataDirs), cifarbinary);
    }
    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto profile = builder->createOptimizationProfile();
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    auto dim = network->getInput(0)->getDimensions();
    dim.d[0] = 1;
    auto dim_min = dim;
    dim_min.d[0] = 1;
    auto dim_max = dim;
    dim_max.d[0]= max_batch_size_;
    auto input_name = network->getInput(0)->getName();
    std::cout<<"dims "<<dim<<endl;

    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, dim_min);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, dim);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, dim_max);

    config->addOptimizationProfile(profile);

    config->setMaxWorkspaceSize(3_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object

    //mEngine->getBindingDimensions(0).d[0] = max_batch_size_;
    //dims.d[0] = max_batch_size_;

    samplesCommon::BufferManager buffers(mEngine,max_batch_size_);

    clock_t total_time = 0;

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    auto input_dims = mEngine->getBindingDimensions(0);
    input_dims.d[0] = max_batch_size_;
    context->setBindingDimensions(0, input_dims);

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    for(int i = 0;i<5;++i){
    if (!processInput(buffers,i))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    
    auto starttime = clock();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    auto endtime = clock();

    sample::gLogInfo << "Use time is :" << double(endtime - starttime) / CLOCKS_PER_SEC << "s" << std::endl;
    
    total_time += (endtime-starttime);

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();


    // Verify results
    if (!verifyOutput(buffers,i))
    {
        return false;
    }

    }

    sample::gLogInfo << "Average Use time is :" << double(total_time) / CLOCKS_PER_SEC / 5 << "s "<<"for batch size "<<max_batch_size_ 
    << std::endl;

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffer,const int batch_idx = 0)
{
    std::cout << "Pre processing begin." << std::endl;
    const int inputC = 3;
    const int inputH = 32;
    const int inputW = 32;
    //const int batchSize = mParams.batchSize;
    const int volImg = inputC * inputH * inputW;
    const int imageSize = volImg + 1;
    const int outputSize = 10;
    float* hostDataBuffer = static_cast<float*>(buffer.getHostBuffer(mParams.inputTensorNames[0]));

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

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffer, const int batch_idx = 0)
{
    const int inputC = 3;
    const int inputH = 32;
    const int inputW = 32;
    //const int batchSize = mParams.batchSize;
    const int volImg = inputC * inputH * inputW;
    const int imageSize = volImg + 1;
    int i = 0;
    float* output = static_cast<float*>(buffer.getHostBuffer(mParams.outputTensorNames[0]));
    int maxposition{0};
    int count{0};
    //for (size_t i = 0; i < 10; i++) {
    //    std::cout << *(output + i) << std::endl;
    //}
    for (size_t i = 0; i < max_batch_size_; i++) {
        maxposition = std::max_element(output+10*i, output+10*i + 10) - (output+10*i);
        //std::cout << "maxposition: " << maxposition << " correctposition: " << int(cifarbinary[(i+32) * imageSize]) << endl;
        if (maxposition == int(cifarbinary[(i + max_batch_size_ * batch_idx) * imageSize])) {
            ++count;
        }
    }
    //std::cout << "The number of correct samples is: " << count << endl;
    float accuracy = float(count) / float(max_batch_size_);
    std::cout << "The accuracy of the TRT Engine on " << max_batch_size_ << " data is: " << accuracy << endl;
    return accuracy;
}


//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("src/sample_vgg16_ori/");
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
        // params.dataDirs.push_back("~/projects/TensorRT-8.0.1.6/samples/cifar10_test");
    }
    params.onnxFileName = "cifar10_vgg16bn_cnn2.onnx";
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.batchSize = args.batch;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    return sample::gLogger.reportPass(sampleTest);
}
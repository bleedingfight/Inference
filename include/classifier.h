#pragma once
#include <argparse/argparse.hpp>

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include <spdlog/spdlog.h>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "base_params.h"
#include "tools.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

class OnnxResnet50 {
  public:
    OnnxResnet50(const ClassifierParams &params);
    bool build();
    bool infer();

  private:
    ClassifierParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.
    nvinfer1::Dims
        mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};  //!< The number to classify

    std::shared_ptr<nvinfer1::IRuntime>
        mRuntime; //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine>
        mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for Resnet50 and creates a TensorRT network
    //!
    bool
    constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                     SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                     SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                     SampleUniquePtr<nvonnxparser::IParser> &parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers);
};

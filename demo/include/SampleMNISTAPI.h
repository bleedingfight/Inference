//
// Created by liushuai on 2020/10/31.
//

#ifndef INFERENCE_SAMPLEMNISTAPI_H
#define INFERENCE_SAMPLEMNISTAPI_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "SampleMNISTAPIParams.h"

class SampleMNISTAPI {
    template<typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMNISTAPI(const SampleMNISTAPIParams &params)
            : mParams(params), mEngine(nullptr) {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleMNISTAPIParams mParams; //!< The parameters for the sample.

    int mNumber{0}; //!< The number to classify

    std::map<std::string, nvinfer1::Weights> mWeightMap; //!< The weight name to weight value map

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Uses the API to create the MNIST Network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                          SampleUniquePtr<nvinfer1::IBuilderConfig> &config);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers);

    //!
    //! \brief Loads weights from weights file
    //!
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string &file);
};


#endif //INFERENCE_SAMPLEMNISTAPI_H

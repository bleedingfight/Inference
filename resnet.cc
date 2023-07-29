// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include <argparse/argparse.hpp>

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <spdlog/spdlog.h>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "tools.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <sstream>
#include <string>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.resnet50";

class OnnxResnet50 {
  public:
    OnnxResnet50(const samplesCommon::OnnxSampleParams &params)
        : mParams(params), mRuntime(nullptr), mEngine(nullptr) {}

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

//!
//! \brief Creates the network, configures the builder and creates the network
//! engine
//!
//! \details This function creates the Onnx Resnet50 network by parsing the Onnx
//! model and builds
//!          the engine that will be used to run Resnet50 (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool OnnxResnet50::build() {
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) {
        return false;
    }
    config->setProfileStream(*profileStream);

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("data", OptProfileSelector::kMIN,
                           Dims4(1, 3, 224, 224));
    profile->setDimensions("data", OptProfileSelector::kOPT,
                           Dims4(1, 3, 224, 224));
    profile->setDimensions("data", OptProfileSelector::kMAX,
                           Dims4(1, 3, 224, 224));
    config->addOptimizationProfile(profile);
    SampleUniquePtr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
        createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime) {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()),
        samplesCommon::InferDeleter());
    if (!mEngine) {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx Resnet50 Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx
//! Resnet50 network
//!
//! \param builder Pointer to the engine builder
//!
bool OnnxResnet50::constructNetwork(
    SampleUniquePtr<nvinfer1::IBuilder> &builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
    SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
    SampleUniquePtr<nvonnxparser::IParser> &parser) {
    auto parsed = parser->parseFromFile(
        mParams.onnxFileName.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    if (mParams.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8) {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It
//! allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool OnnxResnet50::infer() {
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());
    if (!context) {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers)) {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers)) {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool OnnxResnet50::processInput(const samplesCommon::BufferManager &buffers) {
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int N = 3 * 224 * 224;
    float *data = new float[N];
    const string input_np =
        "/home/liushuai/openlab/Inference/inference_trt/blob.bin";

    np_to_vec(input_np, N, data);
    float *hostDataBuffer = static_cast<float *>(
        buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < N; i++) {
        hostDataBuffer[i] = data[i];
    }
    delete[] data;
    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool OnnxResnet50::verifyOutput(const samplesCommon::BufferManager &buffers) {
    const int outputSize = mOutputDims.d[1];
    float *output = static_cast<float *>(
        buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0F};
    int idx{0};

    float sum{0.0F};
    float min_value = std::numeric_limits<float>::min();
    int max_index = -1;
    for (int i = 0; i < outputSize; i++) {
        if (output[i] > min_value) {
            min_value = output[i];
            max_index = i;
        }
    }
    std::cout << "Max value = " << max_index << "\n";
    for (int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i]);
        sum += output[i];
    }
    return max_index == 282;
}

samplesCommon::OnnxSampleParams
initializeSampleParams(const argparse::ArgumentParser &args) {
    samplesCommon::OnnxSampleParams params;
    params.onnxFileName = args.get<string>("onnx");
    params.inputTensorNames.push_back("data");
    params.outputTensorNames.push_back("resnetv17_dense0_fwd");
    params.dlaCore = args.get<int>("useDLACore");
    params.int8 = args.get<bool>("int8");
    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo() {
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or "
                 "--datadir=<path to data directory>] [--useDLACore=<int>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding "
                 "the default. This option can be used "
                 "multiple times to add multiple directories. If no data "
                 "directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
                 "DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char **argv) {
    using namespace std;

    argparse::ArgumentParser program("resnet");
    program.add_argument("-o", "--onnx").help("onnx_model").required();
    program.add_argument("-e", "--engine")
        .help("tensorrt engine model")
        .required();

    program.add_argument("--useDLACore")
        .help("--useDLACore=N  Specify a DLA engine for layers that support "
              "DLA. Value can range from 0 to n-1, ")
        .default_value(-1)
        .implicit_value(true);

    program.add_argument("--input")
        .help("input "
              "f/home/liushuai/openlab/Inference/inference_trt/blob.binilename")
        .default_value("..")
        .implicit_value(true);
    program.add_argument("--int8")
        .help("int8")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--fp16")
        .help("fp16")
        .default_value(false)
        .implicit_value(true);
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &e) {
        std::cerr << e.what() << '\n';
        std::cerr << program;
        return 1;
    }
    auto onnx_filename = program.get<string>("onnx");
    auto engine = program.get<string>("engine");

    spdlog::info("onnx model {}", onnx_filename);
    spdlog::info("engine {}", engine);
    const int N = 3 * 224 * 224;
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    OnnxResnet50 sample(initializeSampleParams(program));

    sample::gLogInfo
        << "Building and running a GPU inference engine for Onnx Resnet50"
        << std::endl;

    if (!sample.build()) {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer()) {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}

#include "classifier.h"
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
    const string input_np = "../datas/blob.bin";

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

OnnxResnet50::OnnxResnet50(const ClassifierParams &params)
    : mParams(params), mRuntime(nullptr), mEngine(nullptr) {}

//
// Created by liushuai on 2020/10/31.
//

#ifndef INFERENCE_UFFNETWORK_H
#define INFERENCE_UFFNETWORK_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvUffParser.h"
#include "NvInfer.h"

class UffNetwork {
    template<typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    UffNetwork(const samplesCommon::UffSampleParams &params)
            : mParams(params) {
    }

    bool build();// 通过Uff文件构建执行引擎
    bool infer(); // 在数据上运行推理引擎
    bool teardown(); // 销毁计算资源

private:
    // 解析Uff文件创建TensorRT网络
    void constructNetwork(
            SampleUniquePtr<nvuffparser::IUffParser> &parser, SampleUniquePtr<nvinfer1::INetworkDefinition> &network);

    // 读取输入和均值数据，预处理，存储结果在buffer里面
    bool processInput(
            const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const;

    // 验证网络输出同时打印
    bool verifyOutput(
            const samplesCommon::BufferManager &buffers, const std::string &outputTensorName,
            int groundTruthDigit) const;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    samplesCommon::UffSampleParams mParams;

    nvinfer1::Dims mInputDims;
    const int kDIGITS{10};
};


#endif //INFERENCE_UFFNETWORK_H

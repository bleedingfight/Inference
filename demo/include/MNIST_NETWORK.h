//
// Created by liushuai on 2020/10/31.
//

#ifndef INFERENCE_MNIST_NETWORK_H
#define INFERENCE_MNIST_NETWORK_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "SampleMNISTAPIParams.h"

class MNIST_NETWORK {
    template<typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    MNIST_NETWORK(const MNIST_NETWORK_Params &params)
            : mParams(params), mEngine(nullptr) {
    }


    bool build();//构建推理网络
    bool infer();//运行TensorRT推理引擎
    bool teardown();//清除引擎相关信息

private:
    MNIST_NETWORK_Params mParams; // 推理需要的参数结构体
    int mNumber{0}; //分类的类别
    std::map<std::string, nvinfer1::Weights> mWeightMap; //权重名称 -> 权重值
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //网络执行的引擎

    // 创建MNIST网络
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                          SampleUniquePtr<nvinfer1::IBuilderConfig> &config);

    // 处理网络的输入，管理存储数据
    bool processInput(const samplesCommon::BufferManager &buffers);

    // 分类数字检查输出
    bool verifyOutput(const samplesCommon::BufferManager &buffers);

    // 从文件中载入权重文件
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string &file);
};

#endif //INFERENCE_MNIST_NETWORK_H

//
// Created by liushuai on 2020/10/31.
//

#ifndef INFERENCE_MNIST_NETWORK_PARAMS_H
#define INFERENCE_MNIST_NETWORK_PARAMS_H

#include "argsParser.h"

struct MNIST_NETWORK_Params : public samplesCommon::SampleParams {
    int inputH;                  //Mnist 输入数据的高度：28
    int inputW;                  //Mnist 输入数据宽度：28
    int outputSize;              //输出数据的Size
    std::string weightsFile;     //输入网络的权重文件
    std::string mnistMeansProto; //!< The proto file containing means
};
#endif //INFERENCE_MNIST_NETWORK_PARAMS_H

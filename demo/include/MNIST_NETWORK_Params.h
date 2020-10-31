//
// Created by liushuai on 2020/10/31.
//

#ifndef INFERENCE_MNIST_NETWORK_PARAMS_H
#define INFERENCE_MNIST_NETWORK_PARAMS_H
#include "argsParser.h"

struct MNIST_NETWORK_Params : public samplesCommon::SampleParams
{
    int inputH;                  //!< The input height
    int inputW;                  //!< The input width
    int outputSize;              //!< The output size
    std::string weightsFile;     //!< The filename of the weights file
    std::string mnistMeansProto; //!< The proto file containing means
};
#endif //INFERENCE_MNIST_NETWORK_PARAMS_H

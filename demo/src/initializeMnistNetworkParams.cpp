//
// Created by liushuai on 2020/10/31.
//

#include "initializeMnistNetworkParams.h"
MNIST_NETWORK_Params initializeMnistSampleParams(const samplesCommon::Args& args)
{
    MNIST_NETWORK_Params params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("/usr/local/TensorRT-7.2.0.14/data/mnist/");
        params.dataDirs.push_back("/usr/local/TensorRT-7.2.0.14/data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.inputH = 28;
    params.inputW = 28;
    params.outputSize = 10;
    params.weightsFile = "mnistapi.wts";
    params.mnistMeansProto = "mnist_mean.binaryproto";

    return params;
}


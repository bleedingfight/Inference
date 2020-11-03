//
// Created by liushuai on 2020/10/31.
//
#include "initializeUffParams.h"
samplesCommon::UffSampleParams initializeUffParams(const samplesCommon::Args& args)
{
    samplesCommon::UffSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided paths
    {
        params.dataDirs.push_back("/usr/local/TensorRT-7.2.0.14/data/mnist/");
        params.dataDirs.push_back("/usr/local/TensorRT-7.2.0.14/data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.uffFileName = locateFile("lenet5.uff", params.dataDirs);
    params.inputTensorNames.push_back("in");
    params.batchSize = 1;
    params.outputTensorNames.push_back("out");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}
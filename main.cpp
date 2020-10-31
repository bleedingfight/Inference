#include "argsParser.h"
#include "logger.h"
#include <cstdlib>
#include <iostream>
#include "MNIST_NETWORK.h"
#include "printHelpInfo.h"
#include "initializeMnistNetworkParams.h"

const std::string gSampleName = "TensorRT.MNIST_Caffe";

int main(int argc, char **argv) {
    samplesCommon::Args args;
    //解析mnist参数
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK) {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help) {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    MNIST_NETWORK sample(initializeMnistSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for MNIST API" << std::endl;

    if (!sample.build()) {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer()) {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown()) {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}

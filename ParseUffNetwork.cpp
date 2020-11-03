//#include "argsParser.h"
//#include "logger.h"
//#include <cstdlib>
//#include <iostream>
//#include <string>
//#include "UffNetworkProcess.h"
//#include "initializeUffParams.h"
//#include "HelpInfo.h"
//const std::string gSampleName = "TensorRT.uff_network";
//
//int main(int argc, char** argv)
//{
//    samplesCommon::Args args;
//    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
//    if (!argsOK)
//    {
//        sample::gLogError << "Invalid arguments" << std::endl;
//        printHelpUffMnistInfo();
//        return EXIT_FAILURE;
//    }
//    if (args.help)
//    {
//        printHelpUffMnistInfo();
//        return EXIT_SUCCESS;
//    }
//    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
//
//    sample::gLogger.reportTestStart(sampleTest);
//
//    samplesCommon::UffSampleParams params = initializeUffParams(args);
//
//    UffNetwork uffNetwork(params);
//    sample::gLogInfo << "Building and running a GPU inference engine for Uff MNIST" << std::endl;
//
//    if (!uffNetwork.build())
//    {
//        return sample::gLogger.reportFail(sampleTest);
//    }
//    if (!uffNetwork.infer())
//    {
//        return sample::gLogger.reportFail(sampleTest);
//    }
//    if (!uffNetwork.teardown())
//    {
//        return sample::gLogger.reportFail(sampleTest);
//    }
//
//    return sample::gLogger.reportPass(sampleTest);
//}
//

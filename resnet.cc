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

#include "base_params.h"
#include "classifier.h"
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
const string gSampleName = "TensorRT:resnet50";
ClassifierParams initializeSampleParams(const argparse::ArgumentParser &args) {
    ClassifierParams params;
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

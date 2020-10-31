/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleMNISTAPI.cpp
//! This file contains the implementation of the MNIST API sample. It creates the network
//! for MNIST classification using the API.
//! It can be run with the following command line:
//! Command: ./sample_mnist_api [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "logger.h"
#include <cstdlib>
#include <iostream>
#include "SampleMNISTAPI.h"
#include "printHelpInfo.h"
#include "initializeMnistSampleParams.h"

const std::string gSampleName = "TensorRT.sample_mnist_api";

int main(int argc, char **argv) {
    samplesCommon::Args args;
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

    SampleMNISTAPI sample(initializeMnistSampleParams(args));

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

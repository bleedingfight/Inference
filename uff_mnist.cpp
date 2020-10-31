//
// Created by liushuai on 2020/10/31.
//
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
//! sampleUffMNIST.cpp
//! This file contains the implementation of the Uff MNIST sample.
//! It creates the network using the MNIST model converted to uff.
//!
//! It can be run with the following command line:
//! Command: ./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "logger.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include "UffNetwork.h"
#include "initializeUffParams.h"
#include "HelpInfo.h"
const std::string gSampleName = "TensorRT.sample_uff_mnist";

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpUffMnistInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpUffMnistInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    samplesCommon::UffSampleParams params = initializeUffParams(args);

    UffNetwork uffNetwork(params);
    sample::gLogInfo << "Building and running a GPU inference engine for Uff MNIST" << std::endl;

    if (!uffNetwork.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!uffNetwork.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!uffNetwork.teardown())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}


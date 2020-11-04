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
//! sampleFasterRCNN_uff.cpp
//! This file contains the implementation of the Uff FasterRCNN sample. It creates the network using
//! the FasterRCNN UFF model.
//! It can be run with the following command line:
//! Command: ./Inference -W 480 -H 272 -I 2016_1111_185016_003_00001_night_000441.ppm
//!

#include "frcnnUtils.h"
#include "logger.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include "UffFasterRcnn.h"

using namespace samplesCommon;

const std::string gSampleName = "TensorRT.uff_fasterRCNN";

int main(int argc, char** argv)
{
    FrcnnArgs args;
    bool argsOK = parseFrcnnArgs(args, argc, argv);

    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    sample::gLogger.reportTestStart(sampleTest);
    auto result = initializeUffRCNNParams(args);
    std::cout<<result.uffFileName<<std::endl;
    UffFasterRcnn uffFasterRcnn(initializeUffRCNNParams(args));
    sample::gLogInfo << "Building and running a GPU inference engine for FasterRCNN" << std::endl;

    if (!uffFasterRcnn.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!uffFasterRcnn.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!uffFasterRcnn.teardown())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}

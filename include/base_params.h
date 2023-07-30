#pragma once
#include <array>
#include <string>
#include <vector>
struct BaseParams {
    int32_t batchSize{1}; //!< Number of inputs in a batch
    int32_t dlaCore{-1};  //!< Specify the DLA core to run network on.
    bool int8{false};     //!< Allow runnning the network in Int8 mode.
    bool fp16{false};     //!< Allow running the network in FP16 mode.
    std::vector<std::string>
        dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::array<std::vector<int>, 3> inputShape;
};
struct ClassifierParams : public BaseParams {
    std::string onnxFileName;
};

//
// Created by liushuai on 2020/7/19.
//

#ifndef TRT_DEMO_BASEPARAMS_H
#define TRT_DEMO_BASEPARAMS_H
#include <vector>
#include <string>
#include <iostream>

class BaseParams
{
    public:
        int batchSize{1};                  //输入数据的Batch size
        int dlaCore{-1};                   //!< Specify the DLA core to run network on.
        bool int8{false};                  //!< 允许网络以 Int8 模式运行.
        bool fp16{false};                  //!< 允许网络以 FP16 模式运行.
        std::vector<std::string> dataDirs; //!< 测试数据的存储路径
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        ~BaseParams();
}

#endif //TRT_DEMO_BASEPARAMS_H

//
// Created by liushuai on 2020/7/19.
//

#ifndef TRT_DEMO_ONNXPARAMS_H
#define TRT_DEMO_ONNXPARAMS_H

#include "BaseParams.h"
class ONNXParams : public BaseParams{
public:
    std::string onnxFileName;
    ~ONNXParams();
};


#endif //TRT_DEMO_ONNXPARAMS_H

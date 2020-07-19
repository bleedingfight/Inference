//
// Created by liushuai on 2020/7/19.
//

#ifndef TRT_DEMO_UFFPARAMS_H
#define TRT_DEMO_UFFPARAMS_H
#include "BaseParams.h"

class UffParams: public BaseParams {
public:
    std::string uffFileName;
    ~UffParams();
};


#endif //TRT_DEMO_UFFPARAMS_H

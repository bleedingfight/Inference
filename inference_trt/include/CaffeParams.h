//
// Created by liushuai on 2020/7/19.
//

#ifndef TRT_DEMO_CAFFEPARAMS_H
#define TRT_DEMO_CAFFEPARAMS_H
#include "BaseParams.h"
class CaffeParams: public BaseParams {
public:
    std::string prototxtFileName; //!< Caffe训练模型的网络的prototxt文件名称
    std::string weightsFileName;  //!< Caffe训练模型的网络的权重文件文件名称
    std::string meanFileName;
    ~CaffeParams();
};
#endif //TRT_DEMO_CAFFEPARAMS_H

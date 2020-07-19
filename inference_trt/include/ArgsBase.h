//
// Created by liushuai on 2020/7/19.
//

#ifndef TRT_DEMO_ARGSBASE_H
#define TRT_DEMO_ARGSBASE_H
#include <vector>
#include <string>
#include <iostream>
class ArgsBase {
public:
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int useDLACore{-1};
    int batch{1};
    std::vector<std::string> dataDirs;
    std::string saveEngine;
    std::string loadEngine;
    bool useILoop{false};
    ~ArgsBase();
};

#endif //TRT_DEMO_ARGSBASE_H

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(logger SHARED logger.cpp)
add_library(getOptions SHARED getOptions.cpp)
add_library(sampleEngines SHARED sampleEngines.cpp)
add_library(sampleInference SHARED sampleInference.cpp)
add_library(sampleOptions SHARED sampleOptions.cpp)
add_library(sampleReporting SHARED sampleReporting.cpp)
add_library(UffParams SHARED UffParams.cpp)
add_library(ArgsBase SHARED ArgsBase.cpp)
add_library(ONNXParams SHARED ONNXParams.cpp)
add_library(CaffeParams SHARED CaffeParams.cpp)

cmake_minimum_required(VERSION 3.16)
project(trt_demo)

set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/local/TensorRT-7.1.3.4/include)
include_directories(include)
set(TENSORRT_LIB_PATH "/usr/local/TensorRT-7.1.3.4/lib")
# 查找TensorRT库文件
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
# CUDA配置
find_package(CUDA 10.2 REQUIRED)
# 调试信息输出
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message("TensorRT_INCLUDE_DIRS:${TENSORRT_LIB_PATH}")
add_subdirectory(src)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(sampleMNISTAPI sampleMNISTAPI.cpp)
target_link_libraries(sampleMNISTAPI ${TRT_LIBS} ${CUDA_LIBRARIES} logger)
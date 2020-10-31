cmake_minimum_required(VERSION 3.16)
project(Inference)

set(CMAKE_CXX_STANDARD 11)
# 设置TensorRT版本
set(TensorRT_VERSION TensorRT-7.2.0.14)
set(TENSORRT_HEAD /usr/local/${TensorRT_VERSION}/include "Galobal variable")
set(CUDA_HEAD "/opt/cuda/targets/x86_64-linux/include" "Galobal variable")
set(TENSORRT_LIB_PATH "/usr/local/${TensorRT_VERSION}/lib")
set(CUDA_LIB_PATH "/opt/cuda/targets/x86_64-linux/lib")
# 包含TensorRT头文件
include_directories(${TENSORRT_HEAD})
# 包含cuda头文件
include_directories(${CUDA_HEAD})

# 包含依赖头文件
include_directories(${CMAKE_SOURCE_DIR}/utils/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找TensorRT库文件
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
file(GLOB CUDA_LIBRARIES "${CUDA_LIB_PATH}/*.so")

set(TRT_LIBS LIBRARIES "Galobal variable")
# CUDA配置
find_package(CUDA 11.0 REQUIRED)
add_subdirectory(utils)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} logger mnistapi)
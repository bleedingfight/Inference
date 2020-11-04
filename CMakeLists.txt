cmake_minimum_required(VERSION 3.16)
project(Inference)

set(CMAKE_CXX_STANDARD 11)
# 设置TensorRT版本
set(TensorRT_VERSION TensorRT-7.2.0.14)
set(TENSORRT_HEAD /usr/local/${TensorRT_VERSION}/include "Galobal variable")
set(CUDA_HEAD "/opt/cuda/targets/x86_64-linux/include" "Galobal variable")

set(TENSORRT_LIB_PATH "/usr/local/${TensorRT_VERSION}/lib")
set(CUDA_LIB_PATH "/opt/cuda/targets/x86_64-linux/lib")
set(DEMO_LIBS demo_lib "Global variable")
# CUDA配置
find_package(CUDA 11.0 REQUIRED)
# 包含TensorRT头文件
include_directories(${TENSORRT_HEAD})
# 包含cuda头文件
include_directories(${CUDA_HEAD})

# 包含依赖头文件
include_directories(${CMAKE_SOURCE_DIR}/utils/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/common/include)
include_directories(${CMAKE_SOURCE_DIR}/demo/include)

# 查找TensorRT库文件
file(GLOB TRT_LIBRARIES "${TENSORRT_LIB_PATH}/*.so")
file(GLOB CUDA_LIBRARIES "${CUDA_LIB_PATH}/*.so")

add_subdirectory(utils)
add_subdirectory(common)
add_subdirectory(demo)
add_subdirectory(plugin)
add_subdirectory(tools)
#add_executable(${PROJECT_NAME} uff_mnist.cpp)
add_executable(${PROJECT_NAME} UffSSD.cpp)

target_link_libraries(${PROJECT_NAME} logger demo_lib common)

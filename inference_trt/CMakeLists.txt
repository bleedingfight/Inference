cmake_minimum_required(VERSION 3.20)
project(trt_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(TENSORRT_HOME /usr/local/TensorRT-8.6.1.6)
include_directories(${TENSORRT_HOME}/include /usr/local/cuda-12.0/targets/x86_64-linux/include include)
set(TENSORRT_LIB_PATH "/usr/local/TensorRT-8.6.1.6/lib")
# 查找TensorRT库文件
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
# CUDA配置
find_package(CUDAToolkit)
find_package(OpenCV)
# 调试信息输出
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message("TensorRT_INCLUDE_DIRS:${TENSORRT_LIB_PATH}")
add_subdirectory(src)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} sampleOnnxMNIST.cpp)
target_link_libraries(${PROJECT_NAME} ${TRT_LIBS} CUDA::cudart utils ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.16)
project(Inference)

set(CMAKE_CXX_STANDARD 11)
# 设置TensorRT版本
set(TensorRT_VERSION TensorRT-7.2.0.14)

# 包含TensorRT头文件
include_directories(/usr/local/${TensorRT_VERSION}/include)
# 包含cuda头文件
include_directories(/opt/cuda/targets/x86_64-linux/include)
# 包含依赖头文件
include_directories(${CMAKE_SOURCE_DIR}/utils/include)
set(TENSORRT_LIB_PATH "/usr/local/${TensorRT_VERSION}/lib")
# 查找TensorRT库文件
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
file(GLOB CUDA_LIBRARIES "/opt/cuda/targets/x86_64-linux/lib/*.so")
# CUDA配置
find_package(CUDA 11.0 REQUIRED)
# 调试信息输出
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
message("TensorRT_INCLUDE_DIRS:${TENSORRT_LIB_PATH}")

add_subdirectory(utils)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${TRT_LIBS} ${CUDA_LIBRARIES} logger)
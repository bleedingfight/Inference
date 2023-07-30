cmake_minimum_required(VERSION 3.20)
project(trt_demo LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
find_package(OpenCV)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

if(NOT TENSORRT_HOME)
  message(FATAL_ERROR "You must define TensorRT Home!(-DTENSORRT_HOME=xxx)")
endif()

if(NOT CUDA_HOME)
  message(FATAL_ERROR "You must define CUDA Home!(-DCUDA_HOME=xxx)")
endif()

set(PROJECT_INCLUDES "")
set(PROJECT_LIBRARIES "")
# 包含 CUDA 和 TensorRT 的头文件
if(CUDA_HOME)
  list(APPEND ${PROJECT_INCLUDES} ${CUDA_HOME}/targets/x86_64-linux/include
       ${TENSORRT_HOME}/include include)
endif()
if(PROJECT_INCLUDES)
  list(APPEND PROJECT_INCLUDES ${TENSORRT_HOME}/targets/x86_64-linux/include)
endif()

add_definitions(-w)
link_directories(${TENSORRT_HOME}/lib)
message("CUDA HOME:${CUDA_HOME}")
message("TensorRT HOME:${TENSORRT_HOME}")
# 是否开启测试
option(BUILD_TESTS "Build the tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
file(GLOB TRT_LIBS "${TENSORRT_HOME}/lib/*.so")
# CUDA配置
include_directories(${CUDA_HOME}/targets/x86_64-linux/include
                    ${TENSORRT_HOME}/include include)
list(APPEND PROJECT_INCLUDES ${TENSORRT_HOME}/include)
add_subdirectory(src)

add_executable(${PROJECT_NAME} resnet.cc)

FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG 997da9255618311d1fcb0135ce86022729d1f1cb)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG 7e635fca68d014934b4af8a1cf874f63989352b7)

FetchContent_MakeAvailable(argparse spdlog)

target_link_libraries(${PROJECT_NAME} CUDA::cudart utils ${OpenCV_LIBS}
                      argparse spdlog::spdlog)

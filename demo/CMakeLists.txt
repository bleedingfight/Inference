include_directories(${CMAKE_SOURCE_DIR}/demo/include)
file(GLOB UTILS_SOURCE ${CMAKE_SOURCE_DIR}/demo/src/*.cpp)
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
file(GLOB CUDA_LIBS "${CUDA_LIB_PATH}/*.so")
add_library(mnistapi SHARED ${UTILS_SOURCE})
target_link_libraries(mnistapi ${TRT_LIBS} ${CUDA_LIBS})


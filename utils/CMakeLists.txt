include_directories(${CMAKE_SOURCE_DIR}/utils/include)
file(GLOB UTILS_SOURCE ${CMAKE_SOURCE_DIR}/utils/src/*.cpp)
#add_library(logger SHARED ${UTILS_SOURCE})
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
file(GLOB CUDA_LIBS "${CUDA_LIB_PATH}/*.so")
add_library(logger SHARED src/logger.cpp)
add_library(mnistapi SHARED src/SampleMNISTAPI.cpp)
target_link_libraries(mnistapi ${TRT_LIBS} ${CUDA_LIBS})


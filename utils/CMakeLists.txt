include_directories(${CMAKE_SOURCE_DIR}/utils/include)
file(GLOB UTILS_SOURCE ${CMAKE_SOURCE_DIR}/utils/src/*.cpp)
add_library(logger SHARED src/logger.cpp)



include_directories(${CMAKE_SOURCE_DIR}/common/include)
file(GLOB COMMON_SRC ${CMAKE_SOURCE_DIR}/common/src/*.cpp)
add_library(common SHARED ${COMMON_SRC})
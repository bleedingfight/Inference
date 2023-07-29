include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRCS "*.cc")
add_library(utils SHARED ${SRCS})
target_link_libraries(utils)

include_directories(${CMAKE_SOURCE_DIR}/host/include ${CMAKE_SOURCE_DIR})

file(GLOB TEST_SRCS "src/*.cc")
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.11.0)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.6.1)
FetchContent_MakeAvailable(googletest googlebenchmark)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

foreach(filename ${TEST_SRCS})
  get_filename_component(TEST_NAME ${filename} NAME_WE)
  set(TARGET_NAME benchmark_${TEST_NAME})
  add_executable(${TARGET_NAME} ${filename})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${TEST_NAME})
  target_link_libraries(${TARGET_NAME} benchmark::benchmark gtest_main utils)
  add_test(${TARGET_NAME} ${TEST_NAME})
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

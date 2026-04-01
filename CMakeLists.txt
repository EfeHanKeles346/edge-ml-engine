cmake_minimum_required(VERSION 3.20)
project(edge_ml_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main library
# add_library(edge_ml
#     src/tensor/tensor.cpp
# )
# target_include_directories(edge_ml PUBLIC include/)

# Main executable
add_executable(edge_ml_engine src/main.cpp)
# target_link_libraries(edge_ml_engine PRIVATE edge_ml)

# Testing
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# add_executable(tensor_test tests/tensor_test.cpp)
# target_link_libraries(tensor_test PRIVATE edge_ml GTest::gtest_main)
# include(GoogleTest)
# gtest_discover_tests(tensor_test)

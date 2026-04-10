cmake_minimum_required(VERSION 3.20)
project(edge_ml_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240116.2
)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v25.3
    GIT_SUBMODULES ""
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)

FetchContent_MakeAvailable(googletest abseil-cpp protobuf)

# Generate ONNX protobuf C++ code
set(ONNX_PROTO ${CMAKE_SOURCE_DIR}/proto/onnx.proto3)
set(ONNX_PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${ONNX_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT ${ONNX_PROTO_GEN_DIR}/onnx.proto3.pb.cc ${ONNX_PROTO_GEN_DIR}/onnx.proto3.pb.h
    COMMAND protoc --cpp_out=${ONNX_PROTO_GEN_DIR} --proto_path=${CMAKE_SOURCE_DIR}/proto ${ONNX_PROTO}
    DEPENDS ${ONNX_PROTO} protoc
    COMMENT "Generating ONNX protobuf C++ code"
)
add_custom_target(onnx_proto_gen DEPENDS ${ONNX_PROTO_GEN_DIR}/onnx.proto3.pb.cc)

# Main library
add_library(edge_ml
    src/tensor/tensor.cpp
    src/operators/activations.cpp
    src/operators/dense.cpp
    src/operators/conv2d.cpp
    src/operators/pooling.cpp
    src/operators/batchnorm.cpp
    src/operators/registry.cpp
    src/parser/onnx_parser.cpp
    src/graph/graph.cpp
    src/graph/executor.cpp
    ${ONNX_PROTO_GEN_DIR}/onnx.proto3.pb.cc
)
add_dependencies(edge_ml onnx_proto_gen)
target_include_directories(edge_ml PUBLIC include/ ${ONNX_PROTO_GEN_DIR})
target_link_libraries(edge_ml PUBLIC libprotobuf)

# Main executable
add_executable(edge_ml_engine src/main.cpp)
target_link_libraries(edge_ml_engine PRIVATE edge_ml)

# Testing
enable_testing()

add_executable(tensor_test tests/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE edge_ml GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(tensor_test)

add_executable(operators_test tests/operators_test.cpp)
target_link_libraries(operators_test PRIVATE edge_ml GTest::gtest_main)
gtest_discover_tests(operators_test)

add_executable(graph_test tests/graph_test.cpp)
target_link_libraries(graph_test PRIVATE edge_ml GTest::gtest_main)
target_compile_definitions(graph_test PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests")
gtest_discover_tests(graph_test)

# Benchmarks
add_executable(matmul_bench benchmarks/matmul_bench.cpp)
target_link_libraries(matmul_bench PRIVATE edge_ml)
target_compile_options(matmul_bench PRIVATE -O2)

# Edge ML Inference Engine

A from-scratch C++ inference engine that loads ONNX models and runs them efficiently on edge devices.

**Goal:** Take a trained ML model (YOLOv11) in ONNX format, parse it, build the computation graph, and execute inference — all written from scratch in C++.

## Why?

Existing engines (PyTorch, TensorFlow, ONNXRuntime) are black boxes. This project builds every layer from the ground up:
- Tensor library with memory management
- Neural network operators (Conv2D, ReLU, MaxPool, BatchNorm...)
- ONNX model parser (protobuf)
- Computation graph with topological sort
- Optimizations: operator fusion, SIMD, quantization

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| [Faz 0](docs/faz-0-kurulum.md) | Project setup: CMake, gtest, protobuf | Not started |
| [Faz 1](docs/faz-1-tensor.md) | Tensor class & memory engine | Not started |
| [Faz 2](docs/faz-2-operatorler.md) | Operator library (Conv2D, ReLU, Pool...) | Not started |
| [Faz 3](docs/faz-3-onnx-graph.md) | ONNX parser & graph executor | Not started |
| [Faz 4](docs/faz-4-optimizasyon.md) | SIMD, quantization, operator fusion | Not started |

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Run Tests

```bash
cd build
ctest --output-on-failure
```

## Project Structure

```
src/
  tensor/       - Tensor class, memory manager
  operators/    - Conv2D, ReLU, MaxPool, BatchNorm, Dense, Softmax
  parser/       - ONNX protobuf parser
  graph/        - Computation graph, topological sort, scheduler
  optimizer/    - Operator fusion, quantization
  engine/       - Main executor, operator registry
include/edge_ml/ - Public headers
tests/          - Google Test unit tests
benchmarks/     - Performance benchmarks
scripts/        - PyTorch validation scripts
examples/       - Usage examples
models/         - ONNX model files
```

## Author

Efe Han Keles — Sabanci University, Computer Science

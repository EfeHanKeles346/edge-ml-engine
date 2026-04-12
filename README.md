# Edge ML Inference Engine

A **from-scratch C++ inference engine** that loads ONNX models and executes them efficiently on edge devices. No PyTorch, no TensorFlow, no ONNXRuntime — every layer built from the ground up.

## What This Engine Does

```
ONNX Model (.onnx) → Parser → Computation Graph → Topological Sort → Executor → Output
```

1. **Reads** an ONNX model file using protobuf
2. **Parses** nodes, weights, and attributes
3. **Builds** a directed acyclic graph (DAG) of operations
4. **Sorts** nodes with topological sort (Kahn's algorithm)
5. **Executes** each operator in order, dispatching via operator registry
6. **Returns** the inference result — validated against ONNXRuntime

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 Edge ML Engine                   │
├─────────┬──────────┬──────────┬─────────────────┤
│ Tensor  │ Operators│  Graph   │   Optimizer      │
│         │          │          │                  │
│ Shape   │ Conv2D   │ Parser   │ SIMD (NEON/SSE) │
│ Stride  │ ReLU     │ Builder  │ Conv+BN Fusion  │
│ Reshape │ MaxPool  │ Topo Sort│ INT8 Quantize   │
│ Matmul  │ BatchNorm│ Executor │ FP16 Simulate   │
│ Add/Mul │ Dense/FC │ Registry │ Benchmarks      │
│         │ Softmax  │          │                  │
│         │ Sigmoid  │          │                  │
│         │ SiLU     │          │                  │
└─────────┴──────────┴──────────┴─────────────────┘
```

## Performance Benchmarks

Tested on Apple Silicon (ARM64 NEON):

### SIMD vs Naive

| Operation | Naive | SIMD | Speedup |
|-----------|-------|------|---------|
| Add (1M elements) | 34.9 ms | 1.6 ms | **21x** |
| Multiply (1M elements) | 34.5 ms | 1.6 ms | **22x** |
| ReLU (1M elements) | 11.6 ms | 1.2 ms | **9.5x** |

### Quantization

| Format | Memory (1M params) | Compression |
|--------|-------------------|-------------|
| FP32 | 3906 KB | 1x |
| INT8 | 976 KB | **4x** |

*Max FP16 precision loss: 0.0005 | Max INT8 quantization error: 0.125*

## Implemented Operators

| Operator | Description | ONNX Name |
|----------|-------------|-----------|
| Conv2D | Naive + im2col | `Conv` |
| ReLU | max(0, x) | `Relu` |
| Sigmoid | 1/(1+exp(-x)) | `Sigmoid` |
| SiLU | x * sigmoid(x) | `Silu` |
| MaxPool2D | Sliding window max | `MaxPool` |
| BatchNorm | Inference mode | `BatchNormalization` |
| Dense/FC | input @ weight + bias | `Gemm` |
| Softmax | Numerically stable | `Softmax` |
| Flatten | N-D to 2D | `Flatten` |
| Reshape | Arbitrary reshape | `Reshape` |
| GlobalAvgPool | Spatial average | `GlobalAveragePool` |
| Add/Sub/Mul | Element-wise | `Add`/`Sub`/`Mul` |
| Transpose | Dimension swap | `Transpose` |

## Build & Run

```bash
# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests (81 tests)
ctest --output-on-failure

# Run benchmarks
./matmul_bench
./optimization_bench

# Run engine
./edge_ml_engine
```

## Project Structure

```
edge-ml-engine/
├── include/edge_ml/
│   ├── tensor.h                    # Core tensor class
│   ├── operators/                  # Neural network operators
│   │   ├── activations.h          # ReLU, Sigmoid, SiLU, Softmax
│   │   ├── conv2d.h               # Conv2D (naive + im2col)
│   │   ├── dense.h                # Fully connected layer
│   │   ├── pooling.h              # MaxPool2D
│   │   ├── batchnorm.h            # Batch normalization
│   │   └── registry.h             # Operator dispatch registry
│   ├── parser/
│   │   └── onnx_parser.h          # ONNX protobuf parser
│   ├── graph/
│   │   ├── graph.h                # DAG + topological sort
│   │   └── executor.h             # Sequential graph executor
│   └── optimizer/
│       ├── simd_ops.h             # NEON/SSE accelerated ops
│       ├── fusion.h               # Conv+BN, Conv+ReLU fusion
│       └── quantization.h         # INT8/FP16 quantization
├── src/                            # Implementations
├── tests/                          # 81 unit tests (Google Test)
├── benchmarks/                     # Performance benchmarks
├── proto/                          # ONNX protobuf schema
└── docs/                           # Phase documentation
```

## Test Coverage

```
81 tests, 0 failures

TensorTest       (35 tests) — shape, stride, reshape, transpose, matmul
OperatorsTest    (27 tests) — all operators validated against known values
GraphTest        (5 tests)  — parser, graph build, topological sort
ExecutorTest     (4 tests)  — end-to-end ONNX model execution
OptimizationTest (10 tests) — SIMD, fusion, quantization correctness
```

## Roadmap

### v1 (Current) — Core Engine

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Project setup: CMake, gtest, protobuf | Done |
| Phase 1 | Tensor class & memory engine | Done |
| Phase 2 | Operator library (13 operators) | Done |
| Phase 3 | ONNX parser & graph executor | Done |
| Phase 4 | SIMD, quantization, operator fusion | Done |

### v2 (Planned) — Real Model Inference

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 5 | Missing operators: Concat, Resize, Upsample, Split, Pad, Clip | Planned |
| Phase 6 | YOLO pipeline: export, inference, bbox decode, NMS | Planned |
| Phase 7 | Multi-threaded executor, memory pool, profiling | Planned |
| Phase 8 | Demo video, benchmarks page, blog post | Planned |

## Technologies

- **C++17** — core implementation
- **CMake** — build system
- **Google Test** — unit testing framework
- **Protocol Buffers** — ONNX model parsing
- **ARM NEON / SSE** — SIMD vectorization
- **GitHub Actions** — CI/CD pipeline

## Author

**Efe Han Keles** — Sabanci University, Computer Science & Engineering

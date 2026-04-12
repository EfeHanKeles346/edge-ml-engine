# Phase 4 — Optimization (3-4 weeks)

## Goal
Speed up the engine: SIMD, operator fusion, quantization, benchmarking.

## SIMD
- [x] Accelerate element-wise ops with SIMD (ARM NEON / SSE)
- [x] Optimize matmul with SIMD
- [x] Benchmark: SIMD vs naive comparison

## Operator Fusion
- [x] Conv + BatchNorm fusion (merge weights)
- [x] Conv + ReLU fusion
- [x] Graph optimizer pass: find fusible nodes and merge them

## Quantization
- [x] FP32 -> FP16 support
- [x] FP32 -> INT8 quantization (scale + zero_point)
- [x] Quantized operators
- [x] Accuracy vs speed comparison

## Benchmark
- [x] Inference time measurement (ms)
- [x] Memory usage
- [x] Before vs after optimization comparison table

## Notes

_Update this section as the phase progresses._

## Related Courses
- CS 303: Cache, SIMD, pipeline

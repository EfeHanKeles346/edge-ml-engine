# Phase 4 — Optimization (3-4 weeks)

## Goal
Speed up the engine: SIMD, operator fusion, quantization, benchmarking.

## SIMD
- [ ] Accelerate element-wise ops with SIMD (ARM NEON / SSE)
- [ ] Optimize matmul with SIMD
- [ ] Benchmark: SIMD vs naive comparison

## Operator Fusion
- [ ] Conv + BatchNorm fusion (merge weights)
- [ ] Conv + ReLU fusion
- [ ] Graph optimizer pass: find fusible nodes and merge them

## Quantization
- [ ] FP32 -> FP16 support
- [ ] FP32 -> INT8 quantization (scale + zero_point)
- [ ] Quantized operators
- [ ] Accuracy vs speed comparison

## Benchmark
- [ ] Inference time measurement (ms)
- [ ] Memory usage
- [ ] Before vs after optimization comparison table

## Notes

_Update this section as the phase progresses._

## Related Courses
- CS 303: Cache, SIMD, pipeline

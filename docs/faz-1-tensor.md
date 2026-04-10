# Phase 1 — Tensor & Memory Engine (2-3 weeks)

## Goal
Build the core data structure: Tensor class with shape, data, stride support. Basic operations (reshape, transpose, element-wise ops, matmul).

## Week 1: Tensor Core
- [x] Tensor class: shape, data (float*), size computation
- [x] Constructor, destructor, copy/move semantics
- [x] operator[] for element access
- [x] Print function (for debugging)
- [x] Test: create 2x3 tensor, assign values, print

## Week 2: Stride & Operations
- [x] Stride support
- [x] reshape (zero-copy, only shape/stride changes)
- [x] transpose (zero-copy, stride swap)
- [x] Element-wise ops: add, subtract, multiply
- [x] Validate against PyTorch

## Week 3: Matrix Multiplication
- [x] Naive matmul (3 nested loops)
- [x] Optimized matmul (loop tiling, cache-friendly)
- [x] Memory alignment (aligned_alloc)
- [x] Benchmark: 1000x1000 matmul in ms
- [x] Validate against PyTorch

## Notes

_Update this section as the phase progresses._

## Related Courses
- CS 201, CS 204: C++ memory management, templates
- MATH 201: Matrix multiplication

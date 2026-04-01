# Phase 1 — Tensor & Memory Engine (2-3 weeks)

## Goal
Build the core data structure: Tensor class with shape, data, stride support. Basic operations (reshape, transpose, element-wise ops, matmul).

## Week 1: Tensor Core
- [ ] Tensor class: shape, data (float*), size computation
- [ ] Constructor, destructor, copy/move semantics
- [ ] operator[] for element access
- [ ] Print function (for debugging)
- [ ] Test: create 2x3 tensor, assign values, print

## Week 2: Stride & Operations
- [ ] Stride support
- [ ] reshape (zero-copy, only shape/stride changes)
- [ ] transpose (zero-copy, stride swap)
- [ ] Element-wise ops: add, subtract, multiply
- [ ] Validate against PyTorch

## Week 3: Matrix Multiplication
- [ ] Naive matmul (3 nested loops)
- [ ] Optimized matmul (loop tiling, cache-friendly)
- [ ] Memory alignment (aligned_alloc)
- [ ] Benchmark: 1000x1000 matmul in ms
- [ ] Validate against PyTorch

## Notes

_Update this section as the phase progresses._

## Related Courses
- CS 201, CS 204: C++ memory management, templates
- MATH 201: Matrix multiplication

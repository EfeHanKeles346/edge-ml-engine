# Phase 2 — Operator Library (3-4 weeks)

## Goal
Implement core neural network layers in C++. Each operator is a separate function/class. Validate all results against PyTorch.

## Week 1: ReLU + Dense (FC)
- [x] ReLU: max(0, x) — element-wise
- [x] Dense: output = input @ weight + bias (uses matmul)
- [x] Validate against PyTorch nn.ReLU and nn.Linear

## Week 2: Conv2D
- [x] Naive Conv2D (5 nested for loops)
- [x] im2col method for Conv2D
- [x] Padding and stride support
- [x] Validate against PyTorch nn.Conv2d

## Week 3: MaxPool, BatchNorm, Sigmoid, SiLU
- [x] MaxPool2D
- [x] BatchNorm (inference formula: gamma*(x-mean)/sqrt(var+eps)+beta)
- [x] Sigmoid: 1/(1+exp(-x))
- [x] SiLU: x * sigmoid(x) — used by YOLOv11
- [x] Validate against PyTorch

## Week 4: Softmax + Registry
- [x] Softmax
- [x] Operator Registry: map<string, function> mapping op_type -> function
- [x] Comprehensive test suite
- [ ] Benchmark all operators

## Notes

_Update this section as the phase progresses._

## Related Courses
- MATH 201: Matrix multiplication (Dense, Conv2D)
- CS 300: Map data structure (Operator Registry)

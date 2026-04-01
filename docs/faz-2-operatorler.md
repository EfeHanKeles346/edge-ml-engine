# Phase 2 — Operator Library (3-4 weeks)

## Goal
Implement core neural network layers in C++. Each operator is a separate function/class. Validate all results against PyTorch.

## Week 1: ReLU + Dense (FC)
- [ ] ReLU: max(0, x) — element-wise
- [ ] Dense: output = input @ weight + bias (uses matmul)
- [ ] Validate against PyTorch nn.ReLU and nn.Linear

## Week 2: Conv2D
- [ ] Naive Conv2D (5 nested for loops)
- [ ] im2col method for Conv2D
- [ ] Padding and stride support
- [ ] Validate against PyTorch nn.Conv2d

## Week 3: MaxPool, BatchNorm, Sigmoid, SiLU
- [ ] MaxPool2D
- [ ] BatchNorm (inference formula: gamma*(x-mean)/sqrt(var+eps)+beta)
- [ ] Sigmoid: 1/(1+exp(-x))
- [ ] SiLU: x * sigmoid(x) — used by YOLOv11
- [ ] Validate against PyTorch

## Week 4: Softmax + Registry
- [ ] Softmax
- [ ] Operator Registry: map<string, function> mapping op_type -> function
- [ ] Comprehensive test suite
- [ ] Benchmark all operators

## Notes

_Update this section as the phase progresses._

## Related Courses
- MATH 201: Matrix multiplication (Dense, Conv2D)
- CS 300: Map data structure (Operator Registry)

# Phase 3 — ONNX Parser & Graph Executor (2-3 weeks)

## Goal
Read ONNX files, build computation graph, sort with topological sort, and execute.

## Week 1: ONNX Parser
- [x] Get onnx.proto3 file (from ONNX GitHub)
- [x] Generate C++ code with protoc
- [x] Open and parse .onnx file
- [x] Print node list (op_type, inputs, outputs)
- [x] Load weights (initializers) into Tensors

## Week 2: Graph Builder
- [x] Graph class: store nodes and edges
- [x] Node connections: which output feeds into which node's input?
- [x] Determine execution order with topological sort
- [x] Load weight tensors into memory

## Week 3: Executor
- [x] Execute sorted nodes sequentially
- [x] For each node: find operator from registry, get input tensors, execute, store output tensor
- [x] Test with simple model (small CNN)
- [x] Validate results against PyTorch/ONNXRuntime

## Notes

_Update this section as the phase progresses._

## Related Courses
- CS 300: Graph, DAG, topological sort

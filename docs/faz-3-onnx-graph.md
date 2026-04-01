# Phase 3 — ONNX Parser & Graph Executor (2-3 weeks)

## Goal
Read ONNX files, build computation graph, sort with topological sort, and execute.

## Week 1: ONNX Parser
- [ ] Get onnx.proto3 file (from ONNX GitHub)
- [ ] Generate C++ code with protoc
- [ ] Open and parse .onnx file
- [ ] Print node list (op_type, inputs, outputs)
- [ ] Load weights (initializers) into Tensors

## Week 2: Graph Builder
- [ ] Graph class: store nodes and edges
- [ ] Node connections: which output feeds into which node's input?
- [ ] Determine execution order with topological sort
- [ ] Load weight tensors into memory

## Week 3: Executor
- [ ] Execute sorted nodes sequentially
- [ ] For each node: find operator from registry, get input tensors, execute, store output tensor
- [ ] Test with simple model (small CNN or MobileNet)
- [ ] Validate results against PyTorch

## Notes

_Update this section as the phase progresses._

## Related Courses
- CS 300: Graph, DAG, topological sort

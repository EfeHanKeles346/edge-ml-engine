# Architecture

## High-Level Flow

```
ONNX File (.onnx)
    |
    v
+---------------+
|  ONNX Parser  |  Read .onnx via protobuf, extract nodes + weights
+-------+-------+
        |
        v
+---------------+
| Graph Builder |  Place nodes with connections into graph
+-------+-------+
        |
        v
+---------------+
|   Optimizer   |  Conv+BN fusion, dead node elimination
+-------+-------+
        |
        v
+---------------+
|   Scheduler   |  Determine execution order via topological sort
+-------+-------+
        |
        v
+---------------+
|   Executor    |  Execute each node in order (find operator from registry)
+-------+-------+
        |
        v
   Output Tensor (result)
```

## Core Components

| Component | Location | Description |
|-----------|----------|-------------|
| Tensor | src/tensor/ | N-dimensional array, memory management, stride |
| Operators | src/operators/ | Conv2D, ReLU, MaxPool, BatchNorm, Dense, Softmax |
| Parser | src/parser/ | ONNX protobuf deserializer |
| Graph | src/graph/ | DAG, topological sort, node connections |
| Optimizer | src/optimizer/ | Fusion, quantization passes |
| Engine | src/engine/ | Executor, operator registry, main entry point |

## Data Flow

```
Input Image (224x224x3)
    -> Tensor [1, 3, 224, 224]
    -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> ... -> Dense -> Softmax
    -> Output Tensor [1, num_classes]
    -> Prediction
```

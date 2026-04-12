#pragma once

#include "edge_ml/tensor.h"
#include <vector>
#include <cstdint>

namespace edge_ml {
namespace opt {

struct QuantizedTensor {
    std::vector<int8_t> data;
    std::vector<int> shape;
    float scale;
    int8_t zero_point;
    int size;
};

// FP32 -> INT8 quantization
QuantizedTensor quantize_int8(const Tensor& input);

// INT8 -> FP32 dequantization
Tensor dequantize_int8(const QuantizedTensor& input);

// Quantized matmul (INT8 inputs, FP32 output)
Tensor quantized_matmul(const Tensor& a, const Tensor& b);

// FP32 -> FP16 -> FP32 simulation (for accuracy comparison)
Tensor simulate_fp16(const Tensor& input);

} // namespace opt
} // namespace edge_ml

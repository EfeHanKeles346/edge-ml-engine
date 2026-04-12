#pragma once

#include "edge_ml/tensor.h"

namespace edge_ml {
namespace opt {

// SIMD-accelerated element-wise operations
Tensor simd_add(const Tensor& a, const Tensor& b);
Tensor simd_multiply(const Tensor& a, const Tensor& b);
Tensor simd_relu(const Tensor& input);

// SIMD-accelerated matrix multiplication
Tensor simd_matmul(const Tensor& a, const Tensor& b);

} // namespace opt
} // namespace edge_ml

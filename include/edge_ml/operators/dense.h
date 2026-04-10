#pragma once

#include "edge_ml/tensor.h"

namespace edge_ml {
namespace ops {

Tensor dense(const Tensor& input, const Tensor& weight, const Tensor& bias);

} // namespace ops
} // namespace edge_ml

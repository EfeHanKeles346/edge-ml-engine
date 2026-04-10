#pragma once

#include "edge_ml/tensor.h"

namespace edge_ml {
namespace ops {

Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor silu(const Tensor& input);
Tensor softmax(const Tensor& input, int axis = -1);

} // namespace ops
} // namespace edge_ml

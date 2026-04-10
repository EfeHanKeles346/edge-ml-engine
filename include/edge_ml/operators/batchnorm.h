#pragma once

#include "edge_ml/tensor.h"

namespace edge_ml {
namespace ops {

Tensor batchnorm(const Tensor& input,
                 const Tensor& gamma, const Tensor& beta,
                 const Tensor& running_mean, const Tensor& running_var,
                 float epsilon = 1e-5f);

} // namespace ops
} // namespace edge_ml

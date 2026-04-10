#pragma once

#include "edge_ml/tensor.h"

namespace edge_ml {
namespace ops {

Tensor maxpool2d(const Tensor& input,
                 int kernel_h, int kernel_w,
                 int stride_h = 1, int stride_w = 1,
                 int pad_h = 0, int pad_w = 0);

} // namespace ops
} // namespace edge_ml

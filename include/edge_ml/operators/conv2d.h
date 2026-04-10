#pragma once

#include "edge_ml/tensor.h"

namespace edge_ml {
namespace ops {

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride_h = 1, int stride_w = 1,
              int pad_h = 0, int pad_w = 0);

Tensor conv2d_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int stride_h = 1, int stride_w = 1,
                     int pad_h = 0, int pad_w = 0);

} // namespace ops
} // namespace edge_ml

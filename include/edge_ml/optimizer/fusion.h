#pragma once

#include "edge_ml/parser/onnx_parser.h"
#include "edge_ml/tensor.h"

namespace edge_ml {
namespace opt {

// Fuse Conv + BatchNorm into a single Conv (merge weights)
void fuse_conv_batchnorm(Tensor& conv_weight, Tensor& conv_bias,
                         const Tensor& bn_gamma, const Tensor& bn_beta,
                         const Tensor& bn_mean, const Tensor& bn_var,
                         float epsilon = 1e-5f);

// Graph-level fusion: modifies the model in-place
int fuse_graph(OnnxModel& model);

} // namespace opt
} // namespace edge_ml

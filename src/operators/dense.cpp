#include "edge_ml/operators/dense.h"
#include <stdexcept>

namespace edge_ml {
namespace ops {

Tensor dense(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: {batch, in_features} or {in_features}
    // weight: {out_features, in_features}
    // bias: {out_features}
    // output: {batch, out_features} or {out_features}

    if (weight.ndim() != 2) {
        throw std::invalid_argument("Dense: weight must be 2D");
    }

    int out_features = weight.shape()[0];
    int in_features = weight.shape()[1];

    if (bias.size() != out_features) {
        throw std::invalid_argument("Dense: bias size must match out_features");
    }

    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

    bool batched = (in.ndim() == 2);
    int batch = batched ? in.shape()[0] : 1;
    int feat = batched ? in.shape()[1] : in.size();

    if (feat != in_features) {
        throw std::invalid_argument("Dense: input features don't match weight");
    }

    // Reshape input to 2D if needed
    Tensor in2d = batched ? in : in.reshape({1, in_features});

    // output = input @ weight^T
    Tensor wt = w.transpose(0, 1);
    Tensor result = in2d.matmul(wt);

    // Add bias to each row
    float* dst = result.data();
    const float* b = bias.data();
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < out_features; j++) {
            dst[i * out_features + j] += b[j];
        }
    }

    if (!batched) {
        return result.reshape({out_features});
    }
    return result;
}

} // namespace ops
} // namespace edge_ml

#include "edge_ml/operators/activations.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace edge_ml {
namespace ops {

Tensor relu(const Tensor& input) {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor output(in.shape());
    const float* src = in.data();
    float* dst = output.data();
    for (int i = 0; i < in.size(); i++) {
        dst[i] = std::max(0.0f, src[i]);
    }
    return output;
}

Tensor sigmoid(const Tensor& input) {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor output(in.shape());
    const float* src = in.data();
    float* dst = output.data();
    for (int i = 0; i < in.size(); i++) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
    return output;
}

Tensor silu(const Tensor& input) {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor output(in.shape());
    const float* src = in.data();
    float* dst = output.data();
    for (int i = 0; i < in.size(); i++) {
        dst[i] = src[i] / (1.0f + std::exp(-src[i]));
    }
    return output;
}

Tensor softmax(const Tensor& input, int axis) {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = in.shape();
    int ndim = in.ndim();

    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw std::invalid_argument("Softmax: invalid axis");
    }

    Tensor output(shape);
    const float* src = in.data();
    float* dst = output.data();

    int outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= shape[i];
    int axis_size = shape[axis];
    int inner_size = 1;
    for (int i = axis + 1; i < ndim; i++) inner_size *= shape[i];

    for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
            int base = o * axis_size * inner_size + i;

            // Find max for numerical stability
            float max_val = src[base];
            for (int a = 1; a < axis_size; a++) {
                float val = src[base + a * inner_size];
                if (val > max_val) max_val = val;
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int a = 0; a < axis_size; a++) {
                float e = std::exp(src[base + a * inner_size] - max_val);
                dst[base + a * inner_size] = e;
                sum += e;
            }

            // Normalize
            for (int a = 0; a < axis_size; a++) {
                dst[base + a * inner_size] /= sum;
            }
        }
    }

    return output;
}

} // namespace ops
} // namespace edge_ml

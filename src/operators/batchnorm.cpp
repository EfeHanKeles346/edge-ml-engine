#include "edge_ml/operators/batchnorm.h"
#include <cmath>
#include <stdexcept>

namespace edge_ml {
namespace ops {

Tensor batchnorm(const Tensor& input,
                 const Tensor& gamma, const Tensor& beta,
                 const Tensor& running_mean, const Tensor& running_var,
                 float epsilon) {
    // input: {batch, channels, H, W}
    // gamma, beta, mean, var: {channels}

    if (input.ndim() != 4) throw std::invalid_argument("BatchNorm: input must be 4D (NCHW)");

    Tensor in = input.is_contiguous() ? input : input.contiguous();

    int batch = in.shape()[0];
    int channels = in.shape()[1];
    int H = in.shape()[2];
    int W = in.shape()[3];

    if (gamma.size() != channels || beta.size() != channels ||
        running_mean.size() != channels || running_var.size() != channels) {
        throw std::invalid_argument("BatchNorm: parameter sizes must match channels");
    }

    // Precompute scale and shift per channel
    // output = scale[c] * x + shift[c]
    // where scale = gamma / sqrt(var + eps), shift = beta - scale * mean
    const float* g = gamma.data();
    const float* b = beta.data();
    const float* mean = running_mean.data();
    const float* var = running_var.data();

    std::vector<float> scale(channels), shift(channels);
    for (int c = 0; c < channels; c++) {
        scale[c] = g[c] / std::sqrt(var[c] + epsilon);
        shift[c] = b[c] - scale[c] * mean[c];
    }

    Tensor output(in.shape());
    const float* in_data = in.data();
    float* out_data = output.data();

    int spatial = H * W;
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            int offset = (n * channels + c) * spatial;
            for (int s = 0; s < spatial; s++) {
                out_data[offset + s] = scale[c] * in_data[offset + s] + shift[c];
            }
        }
    }

    return output;
}

} // namespace ops
} // namespace edge_ml

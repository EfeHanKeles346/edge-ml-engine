#include "edge_ml/operators/pooling.h"
#include <stdexcept>
#include <limits>

namespace edge_ml {
namespace ops {

Tensor maxpool2d(const Tensor& input,
                 int kernel_h, int kernel_w,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w) {
    // input: {batch, channels, in_h, in_w}

    if (input.ndim() != 4) throw std::invalid_argument("MaxPool2D: input must be 4D (NCHW)");

    Tensor in = input.is_contiguous() ? input : input.contiguous();

    int batch = in.shape()[0];
    int channels = in.shape()[1];
    int in_h = in.shape()[2];
    int in_w = in.shape()[3];

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor output({batch, channels, out_h, out_w});
    const float* in_data = in.data();
    float* out_data = output.data();

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                                if (in_data[idx] > max_val) {
                                    max_val = in_data[idx];
                                }
                            }
                        }
                    }

                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = max_val;
                }
            }
        }
    }

    return output;
}

} // namespace ops
} // namespace edge_ml

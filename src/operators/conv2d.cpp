#include "edge_ml/operators/conv2d.h"
#include <stdexcept>
#include <cstring>

namespace edge_ml {
namespace ops {

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride_h, int stride_w, int pad_h, int pad_w) {
    // input:  {batch, in_channels, in_h, in_w}
    // weight: {out_channels, in_channels, kH, kW}
    // bias:   {out_channels}

    if (input.ndim() != 4) throw std::invalid_argument("Conv2D: input must be 4D (NCHW)");
    if (weight.ndim() != 4) throw std::invalid_argument("Conv2D: weight must be 4D");

    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

    int batch = in.shape()[0];
    int in_channels = in.shape()[1];
    int in_h = in.shape()[2];
    int in_w = in.shape()[3];

    int out_channels = w.shape()[0];
    int kH = w.shape()[2];
    int kW = w.shape()[3];

    if (w.shape()[1] != in_channels) {
        throw std::invalid_argument("Conv2D: input channels don't match weight");
    }

    int out_h = (in_h + 2 * pad_h - kH) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kW) / stride_w + 1;

    Tensor output({batch, out_channels, out_h, out_w});
    output.fill(0.0f);

    const float* in_data = in.data();
    const float* w_data = w.data();
    const float* b_data = bias.data();
    float* out_data = output.data();

    for (int n = 0; n < batch; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = b_data[oc];

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_channels + ic) * kH + kh) * kW + kw;
                                    sum += in_data[in_idx] * w_data[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    return output;
}

Tensor conv2d_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int stride_h, int stride_w, int pad_h, int pad_w) {
    // im2col: rearrange input patches into columns, then matmul with reshaped weight

    if (input.ndim() != 4) throw std::invalid_argument("Conv2D im2col: input must be 4D (NCHW)");
    if (weight.ndim() != 4) throw std::invalid_argument("Conv2D im2col: weight must be 4D");

    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

    int batch = in.shape()[0];
    int in_channels = in.shape()[1];
    int in_h = in.shape()[2];
    int in_w = in.shape()[3];

    int out_channels = w.shape()[0];
    int kH = w.shape()[2];
    int kW = w.shape()[3];

    if (w.shape()[1] != in_channels) {
        throw std::invalid_argument("Conv2D im2col: input channels don't match weight");
    }

    int out_h = (in_h + 2 * pad_h - kH) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kW) / stride_w + 1;

    int col_rows = in_channels * kH * kW;
    int col_cols = out_h * out_w;

    // Reshape weight to 2D: {out_channels, in_channels * kH * kW}
    Tensor w_reshaped = w.reshape({out_channels, col_rows});

    Tensor output({batch, out_channels, out_h, out_w});
    const float* in_data = in.data();
    const float* b_data = bias.data();
    float* out_data = output.data();

    // Process each batch
    for (int n = 0; n < batch; n++) {
        // Build im2col matrix: {col_rows, col_cols}
        Tensor columns({col_rows, col_cols});
        float* col_data = columns.data();
        std::memset(col_data, 0, col_rows * col_cols * sizeof(float));

        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int col_row = (ic * kH + kh) * kW + kw;
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            int col_col = oh * out_w + ow;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = (n * in_channels + ic) * in_h * in_w + ih * in_w + iw;
                                col_data[col_row * col_cols + col_col] = in_data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        // Matmul: {out_channels, col_rows} x {col_rows, col_cols} = {out_channels, col_cols}
        Tensor result = w_reshaped.matmul(columns);
        const float* res_data = result.data();

        // Copy result + add bias
        for (int oc = 0; oc < out_channels; oc++) {
            for (int p = 0; p < col_cols; p++) {
                int out_idx = (n * out_channels + oc) * col_cols + p;
                out_data[out_idx] = res_data[oc * col_cols + p] + b_data[oc];
            }
        }
    }

    return output;
}

} // namespace ops
} // namespace edge_ml

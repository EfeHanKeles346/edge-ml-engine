#include "edge_ml/optimizer/quantization.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace edge_ml {
namespace opt {

QuantizedTensor quantize_int8(const Tensor& input) {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    const float* data = in.data();
    int n = in.size();

    // Find min and max
    float min_val = data[0], max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    // Compute scale and zero_point
    // Maps [min_val, max_val] -> [-128, 127]
    float scale = (max_val - min_val) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;

    float zero_point_f = -128.0f - min_val / scale;
    int8_t zero_point = static_cast<int8_t>(
        std::max(-128.0f, std::min(127.0f, std::round(zero_point_f)))
    );

    QuantizedTensor qt;
    qt.scale = scale;
    qt.zero_point = zero_point;
    qt.shape = in.shape();
    qt.size = n;
    qt.data.resize(n);

    for (int i = 0; i < n; i++) {
        float q = std::round(data[i] / scale) + zero_point;
        q = std::max(-128.0f, std::min(127.0f, q));
        qt.data[i] = static_cast<int8_t>(q);
    }

    return qt;
}

Tensor dequantize_int8(const QuantizedTensor& input) {
    Tensor output(input.shape);
    float* data = output.data();

    for (int i = 0; i < input.size; i++) {
        data[i] = (static_cast<float>(input.data[i]) - input.zero_point) * input.scale;
    }

    return output;
}

Tensor quantized_matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("quantized_matmul: inputs must be 2D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("quantized_matmul: dimension mismatch");
    }

    // Quantize inputs
    QuantizedTensor qa = quantize_int8(a);
    QuantizedTensor qb = quantize_int8(b);

    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];

    // INT8 matmul with INT32 accumulation
    Tensor output({M, N});
    float* out = output.data();
    float combined_scale = qa.scale * qb.scale;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int32_t va = static_cast<int32_t>(qa.data[i * K + k]) - qa.zero_point;
                int32_t vb = static_cast<int32_t>(qb.data[k * N + j]) - qb.zero_point;
                acc += va * vb;
            }
            out[i * N + j] = static_cast<float>(acc) * combined_scale;
        }
    }

    return output;
}

Tensor simulate_fp16(const Tensor& input) {
    // Simulate FP16 by reducing precision
    // FP16 has 10-bit mantissa (vs FP32's 23-bit)
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor output(in.shape());

    const float* src = in.data();
    float* dst = output.data();

    for (int i = 0; i < in.size(); i++) {
        // Convert to half precision and back
        // This simulates the precision loss of FP16
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));

        // Zero out the lower 13 bits of mantissa (23 - 10 = 13)
        bits &= 0xFFFFE000u;

        std::memcpy(&dst[i], &bits, sizeof(float));
    }

    return output;
}

} // namespace opt
} // namespace edge_ml

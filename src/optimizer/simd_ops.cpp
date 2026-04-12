#include "edge_ml/optimizer/simd_ops.h"
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#elif defined(__SSE__)
#include <xmmintrin.h>
#define HAS_SSE 1
#endif

namespace edge_ml {
namespace opt {

Tensor simd_add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("simd_add: shape mismatch");
    }

    Tensor in_a = a.is_contiguous() ? a : a.contiguous();
    Tensor in_b = b.is_contiguous() ? b : b.contiguous();
    Tensor output(in_a.shape());

    const float* src_a = in_a.data();
    const float* src_b = in_b.data();
    float* dst = output.data();
    int n = in_a.size();
    int i = 0;

#if defined(HAS_NEON)
    // Process 4 floats at a time with NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(src_a + i);
        float32x4_t vb = vld1q_f32(src_b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(dst + i, vr);
    }
#elif defined(HAS_SSE)
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(src_a + i);
        __m128 vb = _mm_loadu_ps(src_b + i);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(dst + i, vr);
    }
#endif

    // Scalar fallback for remaining elements
    for (; i < n; i++) {
        dst[i] = src_a[i] + src_b[i];
    }

    return output;
}

Tensor simd_multiply(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("simd_multiply: shape mismatch");
    }

    Tensor in_a = a.is_contiguous() ? a : a.contiguous();
    Tensor in_b = b.is_contiguous() ? b : b.contiguous();
    Tensor output(in_a.shape());

    const float* src_a = in_a.data();
    const float* src_b = in_b.data();
    float* dst = output.data();
    int n = in_a.size();
    int i = 0;

#if defined(HAS_NEON)
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(src_a + i);
        float32x4_t vb = vld1q_f32(src_b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(dst + i, vr);
    }
#elif defined(HAS_SSE)
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(src_a + i);
        __m128 vb = _mm_loadu_ps(src_b + i);
        __m128 vr = _mm_mul_ps(va, vb);
        _mm_storeu_ps(dst + i, vr);
    }
#endif

    for (; i < n; i++) {
        dst[i] = src_a[i] * src_b[i];
    }

    return output;
}

Tensor simd_relu(const Tensor& input) {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor output(in.shape());

    const float* src = in.data();
    float* dst = output.data();
    int n = in.size();
    int i = 0;

#if defined(HAS_NEON)
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float32x4_t vr = vmaxq_f32(v, vzero);
        vst1q_f32(dst + i, vr);
    }
#elif defined(HAS_SSE)
    __m128 vzero = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 v = _mm_loadu_ps(src + i);
        __m128 vr = _mm_max_ps(v, vzero);
        _mm_storeu_ps(dst + i, vr);
    }
#endif

    for (; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }

    return output;
}

Tensor simd_matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("simd_matmul: inputs must be 2D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("simd_matmul: dimension mismatch");
    }

    Tensor in_a = a.is_contiguous() ? a : a.contiguous();
    Tensor in_b = b.is_contiguous() ? b : b.contiguous();

    int M = in_a.shape()[0];
    int K = in_a.shape()[1];
    int N = in_b.shape()[0 + 1];

    Tensor output({M, N});
    output.fill(0.0f);

    const float* A = in_a.data();
    const float* B = in_b.data();
    float* C = output.data();

    // Tiled + SIMD matmul
    const int TILE = 32;

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int i_end = (i0 + TILE < M) ? i0 + TILE : M;
                int j_end = (j0 + TILE < N) ? j0 + TILE : N;
                int k_end = (k0 + TILE < K) ? k0 + TILE : K;

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * K + k];
                        int j = j0;

#if defined(HAS_NEON)
                        float32x4_t va = vdupq_n_f32(a_ik);
                        for (; j + 3 < j_end; j += 4) {
                            float32x4_t vb = vld1q_f32(B + k * N + j);
                            float32x4_t vc = vld1q_f32(C + i * N + j);
                            vc = vmlaq_f32(vc, va, vb); // vc += va * vb
                            vst1q_f32(C + i * N + j, vc);
                        }
#elif defined(HAS_SSE)
                        __m128 va = _mm_set1_ps(a_ik);
                        for (; j + 3 < j_end; j += 4) {
                            __m128 vb = _mm_loadu_ps(B + k * N + j);
                            __m128 vc = _mm_loadu_ps(C + i * N + j);
                            vc = _mm_add_ps(vc, _mm_mul_ps(va, vb));
                            _mm_storeu_ps(C + i * N + j, vc);
                        }
#endif
                        // Scalar remainder
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }

    return output;
}

} // namespace opt
} // namespace edge_ml

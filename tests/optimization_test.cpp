#include <gtest/gtest.h>
#include "edge_ml/tensor.h"
#include "edge_ml/operators/activations.h"
#include "edge_ml/operators/conv2d.h"
#include "edge_ml/operators/batchnorm.h"
#include "edge_ml/optimizer/simd_ops.h"
#include "edge_ml/optimizer/fusion.h"
#include "edge_ml/optimizer/quantization.h"
#include <cmath>

using namespace edge_ml;
using namespace edge_ml::ops;
using namespace edge_ml::opt;

// ==================== SIMD Tests ====================

TEST(SimdTest, AddMatchesNaive) {
    Tensor a({100}, std::vector<float>(100));
    Tensor b({100}, std::vector<float>(100));
    for (int i = 0; i < 100; i++) {
        a[i] = (float)i * 0.1f;
        b[i] = (float)(100 - i) * 0.2f;
    }

    Tensor naive = a.add(b);
    Tensor simd = simd_add(a, b);

    for (int i = 0; i < 100; i++) {
        EXPECT_NEAR(naive[i], simd[i], 1e-5);
    }
}

TEST(SimdTest, MultiplyMatchesNaive) {
    Tensor a({100}, std::vector<float>(100));
    Tensor b({100}, std::vector<float>(100));
    for (int i = 0; i < 100; i++) {
        a[i] = (float)i * 0.1f - 5.0f;
        b[i] = (float)(i % 7) * 0.3f;
    }

    Tensor naive = a.multiply(b);
    Tensor simd = simd_multiply(a, b);

    for (int i = 0; i < 100; i++) {
        EXPECT_NEAR(naive[i], simd[i], 1e-5);
    }
}

TEST(SimdTest, ReluMatchesNaive) {
    Tensor a({50}, std::vector<float>(50));
    for (int i = 0; i < 50; i++) {
        a[i] = (float)i - 25.0f;
    }

    Tensor naive = relu(a);
    Tensor simd = simd_relu(a);

    for (int i = 0; i < 50; i++) {
        EXPECT_NEAR(naive[i], simd[i], 1e-5);
    }
}

TEST(SimdTest, MatmulMatchesNaive) {
    Tensor a({16, 20}, std::vector<float>(320));
    Tensor b({20, 12}, std::vector<float>(240));
    for (int i = 0; i < 320; i++) a[i] = (float)(i % 11) * 0.1f - 0.5f;
    for (int i = 0; i < 240; i++) b[i] = (float)(i % 7) * 0.2f - 0.7f;

    Tensor naive = a.matmul(b);
    Tensor simd = simd_matmul(a, b);

    EXPECT_EQ(naive.shape(), simd.shape());
    for (int i = 0; i < naive.size(); i++) {
        EXPECT_NEAR(naive[i], simd[i], 1e-3);
    }
}

TEST(SimdTest, AddOddSize) {
    // Test with size not divisible by 4 (SIMD lane width)
    Tensor a({7}, {1, 2, 3, 4, 5, 6, 7});
    Tensor b({7}, {7, 6, 5, 4, 3, 2, 1});

    Tensor result = simd_add(a, b);
    for (int i = 0; i < 7; i++) {
        EXPECT_FLOAT_EQ(result[i], 8.0f);
    }
}

// ==================== Fusion Tests ====================

TEST(FusionTest, ConvBatchNormFusion) {
    // Create Conv weights
    Tensor conv_w({2, 1, 3, 3});
    for (int i = 0; i < conv_w.size(); i++) conv_w[i] = 0.5f;
    Tensor conv_b({2}, {0.0f, 0.0f});

    // Create BN parameters
    Tensor gamma({2}, {1.0f, 2.0f});
    Tensor beta({2}, {0.1f, 0.2f});
    Tensor mean({2}, {0.5f, 0.5f});
    Tensor var({2}, {1.0f, 1.0f});

    // Create input
    Tensor input({1, 1, 5, 5});
    input.fill(1.0f);

    // Unfused: Conv then BatchNorm
    Tensor conv_out = conv2d(input, conv_w, conv_b);
    Tensor unfused = batchnorm(conv_out, gamma, beta, mean, var);

    // Fuse and run just Conv
    fuse_conv_batchnorm(conv_w, conv_b, gamma, beta, mean, var);
    Tensor fused = conv2d(input, conv_w, conv_b);

    // Results should match
    EXPECT_EQ(fused.shape(), unfused.shape());
    for (int i = 0; i < fused.size(); i++) {
        EXPECT_NEAR(fused[i], unfused[i], 1e-4);
    }
}

// ==================== Quantization Tests ====================

TEST(QuantizationTest, QuantizeDequantize) {
    Tensor t({5}, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});

    auto qt = quantize_int8(t);
    EXPECT_EQ(qt.size, 5);
    EXPECT_EQ(qt.shape, (std::vector<int>{5}));

    Tensor recovered = dequantize_int8(qt);
    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(t[i], recovered[i], 0.02f); // INT8 has limited precision
    }
}

TEST(QuantizationTest, QuantizedMatmul) {
    Tensor a({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor b({3, 2}, {1, 2, 3, 4, 5, 6});

    Tensor fp32_result = a.matmul(b);
    Tensor int8_result = quantized_matmul(a, b);

    EXPECT_EQ(fp32_result.shape(), int8_result.shape());

    // INT8 matmul should be reasonably close to FP32
    for (int i = 0; i < fp32_result.size(); i++) {
        float rel_error = std::abs(fp32_result[i] - int8_result[i]) /
                          (std::abs(fp32_result[i]) + 1e-6f);
        EXPECT_LT(rel_error, 0.15f); // within 15% relative error
    }
}

TEST(QuantizationTest, FP16Simulation) {
    Tensor t({4}, {1.0f, 0.1f, 0.001f, 100.0f});
    Tensor fp16 = simulate_fp16(t);

    // FP16 should preserve large values well
    EXPECT_NEAR(fp16[0], 1.0f, 0.001f);
    EXPECT_NEAR(fp16[3], 100.0f, 0.1f);
}

TEST(QuantizationTest, MemoryReduction) {
    Tensor t({1000});
    t.fill(1.0f);

    auto qt = quantize_int8(t);

    // INT8 should be 4x smaller
    size_t fp32_bytes = 1000 * sizeof(float);
    size_t int8_bytes = qt.data.size() * sizeof(int8_t);
    EXPECT_EQ(fp32_bytes / int8_bytes, 4u);
}

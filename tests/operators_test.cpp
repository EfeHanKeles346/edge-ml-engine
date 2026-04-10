#include <gtest/gtest.h>
#include "edge_ml/tensor.h"
#include "edge_ml/operators/activations.h"
#include "edge_ml/operators/dense.h"
#include "edge_ml/operators/conv2d.h"
#include "edge_ml/operators/pooling.h"
#include "edge_ml/operators/batchnorm.h"
#include "edge_ml/operators/registry.h"
#include <cmath>

using namespace edge_ml;
using namespace edge_ml::ops;

// ==================== ReLU Tests ====================

TEST(ReLUTest, Basic) {
    Tensor t({5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    Tensor out = relu(t);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
    EXPECT_FLOAT_EQ(out[4], 2.0f);
}

TEST(ReLUTest, AllNegative) {
    Tensor t({3}, {-5.0f, -3.0f, -1.0f});
    Tensor out = relu(t);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}

TEST(ReLUTest, MultiDim) {
    Tensor t({2, 3}, {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f});
    Tensor out = relu(t);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
    EXPECT_FLOAT_EQ(out[4], 0.0f);
    EXPECT_FLOAT_EQ(out[5], 6.0f);
}

// ==================== Sigmoid Tests ====================

TEST(SigmoidTest, Basic) {
    Tensor t({3}, {0.0f, 1.0f, -1.0f});
    Tensor out = sigmoid(t);
    EXPECT_NEAR(out[0], 0.5f, 1e-5);
    EXPECT_NEAR(out[1], 0.7310586f, 1e-5);
    EXPECT_NEAR(out[2], 0.2689414f, 1e-5);
}

TEST(SigmoidTest, LargeValues) {
    Tensor t({2}, {100.0f, -100.0f});
    Tensor out = sigmoid(t);
    EXPECT_NEAR(out[0], 1.0f, 1e-5);
    EXPECT_NEAR(out[1], 0.0f, 1e-5);
}

// ==================== SiLU Tests ====================

TEST(SiLUTest, Basic) {
    Tensor t({3}, {0.0f, 1.0f, -1.0f});
    Tensor out = silu(t);
    // SiLU(x) = x * sigmoid(x)
    EXPECT_NEAR(out[0], 0.0f, 1e-5);
    EXPECT_NEAR(out[1], 1.0f * 0.7310586f, 1e-5);
    EXPECT_NEAR(out[2], -1.0f * 0.2689414f, 1e-5);
}

// ==================== Softmax Tests ====================

TEST(SoftmaxTest, Basic1D) {
    Tensor t({3}, {1.0f, 2.0f, 3.0f});
    Tensor out = softmax(t, 0);
    // Check sums to 1
    float sum = out[0] + out[1] + out[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5);
    // Check ordering: out[2] > out[1] > out[0]
    EXPECT_GT(out[2], out[1]);
    EXPECT_GT(out[1], out[0]);
}

TEST(SoftmaxTest, Basic2D) {
    Tensor t({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    Tensor out = softmax(t, -1); // softmax along last axis
    // Each row should sum to 1
    float sum0 = out.at({0, 0}) + out.at({0, 1}) + out.at({0, 2});
    float sum1 = out.at({1, 0}) + out.at({1, 1}) + out.at({1, 2});
    EXPECT_NEAR(sum0, 1.0f, 1e-5);
    EXPECT_NEAR(sum1, 1.0f, 1e-5);
    // Equal inputs -> equal outputs
    EXPECT_NEAR(out.at({1, 0}), out.at({1, 1}), 1e-5);
    EXPECT_NEAR(out.at({1, 1}), out.at({1, 2}), 1e-5);
}

TEST(SoftmaxTest, NumericalStability) {
    // Large values should not cause overflow
    Tensor t({3}, {1000.0f, 1001.0f, 1002.0f});
    Tensor out = softmax(t, 0);
    float sum = out[0] + out[1] + out[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5);
}

// ==================== Dense Tests ====================

TEST(DenseTest, Basic) {
    // input: {1, 3}, weight: {2, 3}, bias: {2}
    // output = input @ weight^T + bias
    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor weight({2, 3}, {1.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f});
    Tensor bias({2}, {0.5f, -0.5f});
    Tensor out = dense(input, weight, bias);

    EXPECT_EQ(out.shape()[0], 1);
    EXPECT_EQ(out.shape()[1], 2);
    // out[0] = 1*1 + 2*0 + 3*0 + 0.5 = 1.5
    // out[1] = 1*0 + 2*1 + 3*0 - 0.5 = 1.5
    EXPECT_NEAR(out.at({0, 0}), 1.5f, 1e-5);
    EXPECT_NEAR(out.at({0, 1}), 1.5f, 1e-5);
}

TEST(DenseTest, Unbatched) {
    Tensor input({3}, {1.0f, 2.0f, 3.0f});
    Tensor weight({2, 3}, {1.0f, 1.0f, 1.0f,
                           2.0f, 2.0f, 2.0f});
    Tensor bias({2}, {0.0f, 1.0f});
    Tensor out = dense(input, weight, bias);

    EXPECT_EQ(out.ndim(), 1);
    EXPECT_EQ(out.size(), 2);
    // out[0] = 1+2+3 + 0 = 6
    // out[1] = 2+4+6 + 1 = 13
    EXPECT_NEAR(out[0], 6.0f, 1e-5);
    EXPECT_NEAR(out[1], 13.0f, 1e-5);
}

TEST(DenseTest, BatchedMultiple) {
    Tensor input({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor weight({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor bias({3}, {0.0f, 0.0f, 0.0f});
    Tensor out = dense(input, weight, bias);

    EXPECT_EQ(out.shape()[0], 2);
    EXPECT_EQ(out.shape()[1], 3);
    // Row 0 (input=[1,0]): out = [1, 3, 5]
    // Row 1 (input=[0,1]): out = [2, 4, 6]
    EXPECT_NEAR(out.at({0, 0}), 1.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 1}), 3.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 2}), 5.0f, 1e-5);
    EXPECT_NEAR(out.at({1, 0}), 2.0f, 1e-5);
    EXPECT_NEAR(out.at({1, 1}), 4.0f, 1e-5);
    EXPECT_NEAR(out.at({1, 2}), 6.0f, 1e-5);
}

// ==================== Conv2D Tests ====================

TEST(Conv2DTest, NaiveBasic) {
    // 1x1x3x3 input, 1x1x2x2 kernel
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor weight({1, 1, 2, 2}, {1, 0, 0, 1});
    Tensor bias({1}, {0.0f});

    Tensor out = conv2d(input, weight, bias);

    EXPECT_EQ(out.shape()[2], 2); // out_h
    EXPECT_EQ(out.shape()[3], 2); // out_w
    // out[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // out[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // out[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // out[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    EXPECT_NEAR(out.at({0, 0, 0, 0}), 6.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 0, 0, 1}), 8.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 0, 1, 0}), 12.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 0, 1, 1}), 14.0f, 1e-5);
}

TEST(Conv2DTest, WithPadding) {
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor weight({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0}); // identity kernel
    Tensor bias({1}, {0.0f});

    Tensor out = conv2d(input, weight, bias, 1, 1, 1, 1); // pad=1
    EXPECT_EQ(out.shape()[2], 3);
    EXPECT_EQ(out.shape()[3], 3);
    // Identity kernel with pad=1 should reproduce input
    EXPECT_NEAR(out.at({0, 0, 0, 0}), 1.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 0, 1, 1}), 5.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 0, 2, 2}), 9.0f, 1e-5);
}

TEST(Conv2DTest, WithStride) {
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; i++) input[i] = (float)(i + 1);
    Tensor weight({1, 1, 2, 2}, {1, 1, 1, 1}); // sum kernel
    Tensor bias({1}, {0.0f});

    Tensor out = conv2d(input, weight, bias, 2, 2); // stride=2
    EXPECT_EQ(out.shape()[2], 2);
    EXPECT_EQ(out.shape()[3], 2);
}

TEST(Conv2DTest, WithBias) {
    Tensor input({1, 1, 2, 2}, {1, 1, 1, 1});
    Tensor weight({1, 1, 1, 1}, {2.0f});
    Tensor bias({1}, {3.0f});

    Tensor out = conv2d(input, weight, bias);
    // Each output = 2*1 + 3 = 5
    for (int i = 0; i < out.size(); i++) {
        EXPECT_NEAR(out[i], 5.0f, 1e-5);
    }
}

TEST(Conv2DTest, Im2colMatchesNaive) {
    // Both methods should produce same results
    Tensor input({1, 2, 4, 4});
    for (int i = 0; i < input.size(); i++) input[i] = (float)(i % 7) - 3.0f;
    Tensor weight({3, 2, 3, 3});
    for (int i = 0; i < weight.size(); i++) weight[i] = (float)(i % 5) * 0.1f;
    Tensor bias({3}, {0.1f, -0.2f, 0.3f});

    Tensor naive_out = conv2d(input, weight, bias, 1, 1, 1, 1);
    Tensor im2col_out = conv2d_im2col(input, weight, bias, 1, 1, 1, 1);

    EXPECT_EQ(naive_out.shape(), im2col_out.shape());
    for (int i = 0; i < naive_out.size(); i++) {
        EXPECT_NEAR(naive_out[i], im2col_out[i], 1e-4);
    }
}

TEST(Conv2DTest, MultipleChannels) {
    // 2 input channels, 2 output channels
    Tensor input({1, 2, 3, 3});
    for (int i = 0; i < 18; i++) input[i] = 1.0f;
    Tensor weight({2, 2, 2, 2});
    for (int i = 0; i < 16; i++) weight[i] = 0.5f;
    Tensor bias({2}, {0.0f, 1.0f});

    Tensor out = conv2d(input, weight, bias);
    EXPECT_EQ(out.shape()[0], 1);
    EXPECT_EQ(out.shape()[1], 2);
    EXPECT_EQ(out.shape()[2], 2);
    EXPECT_EQ(out.shape()[3], 2);
    // Each output = sum of 2*2*2 weights * 1.0 + bias
    // = 8 * 0.5 + bias = 4.0 + bias
    EXPECT_NEAR(out.at({0, 0, 0, 0}), 4.0f, 1e-5);
    EXPECT_NEAR(out.at({0, 1, 0, 0}), 5.0f, 1e-5);
}

// ==================== MaxPool2D Tests ====================

TEST(MaxPool2DTest, Basic) {
    Tensor input({1, 1, 4, 4}, {
        1, 3, 2, 1,
        4, 6, 5, 3,
        4, 2, 1, 0,
        1, 2, 5, 4
    });

    Tensor out = maxpool2d(input, 2, 2, 2, 2);
    EXPECT_EQ(out.shape()[2], 2);
    EXPECT_EQ(out.shape()[3], 2);
    EXPECT_FLOAT_EQ(out.at({0, 0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 1, 1}), 5.0f);
}

TEST(MaxPool2DTest, Stride1) {
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor out = maxpool2d(input, 2, 2, 1, 1);
    EXPECT_EQ(out.shape()[2], 2);
    EXPECT_EQ(out.shape()[3], 2);
    EXPECT_FLOAT_EQ(out.at({0, 0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 1, 0}), 8.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 1, 1}), 9.0f);
}

TEST(MaxPool2DTest, MultiChannel) {
    Tensor input({1, 2, 4, 4});
    for (int i = 0; i < 16; i++) input[i] = (float)i;       // channel 0
    for (int i = 0; i < 16; i++) input[16 + i] = (float)(15 - i); // channel 1 (reversed)

    Tensor out = maxpool2d(input, 2, 2, 2, 2);
    EXPECT_EQ(out.shape()[1], 2);
    // Channel 0: max of {0,1,4,5}=5, {2,3,6,7}=7, {8,9,12,13}=13, {10,11,14,15}=15
    EXPECT_FLOAT_EQ(out.at({0, 0, 0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(out.at({0, 0, 1, 1}), 15.0f);
}

// ==================== BatchNorm Tests ====================

TEST(BatchNormTest, Basic) {
    // 1 batch, 2 channels, 2x2 spatial
    Tensor input({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor gamma({2}, {1.0f, 1.0f});
    Tensor beta({2}, {0.0f, 0.0f});
    Tensor mean({2}, {2.5f, 6.5f});
    Tensor var({2}, {1.25f, 1.25f});

    Tensor out = batchnorm(input, gamma, beta, mean, var);
    EXPECT_EQ(out.shape(), input.shape());

    // Channel 0: (x - 2.5) / sqrt(1.25 + 1e-5)
    float scale = 1.0f / std::sqrt(1.25f + 1e-5f);
    EXPECT_NEAR(out.at({0, 0, 0, 0}), (1.0f - 2.5f) * scale, 1e-4);
    EXPECT_NEAR(out.at({0, 0, 1, 1}), (4.0f - 2.5f) * scale, 1e-4);
}

TEST(BatchNormTest, WithGammaBeta) {
    Tensor input({1, 1, 2, 2}, {0, 0, 0, 0});
    Tensor gamma({1}, {2.0f});
    Tensor beta({1}, {3.0f});
    Tensor mean({1}, {0.0f});
    Tensor var({1}, {1.0f});

    Tensor out = batchnorm(input, gamma, beta, mean, var);
    // output = 2.0 * (0 - 0) / sqrt(1 + 1e-5) + 3.0 ≈ 3.0
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(out[i], 3.0f, 1e-4);
    }
}

// ==================== Registry Tests ====================

TEST(RegistryTest, RegisterAndDispatch) {
    register_all_ops();
    auto& reg = OpRegistry::instance();

    EXPECT_TRUE(reg.has_op("Relu"));
    EXPECT_TRUE(reg.has_op("Sigmoid"));
    EXPECT_TRUE(reg.has_op("Conv"));
    EXPECT_TRUE(reg.has_op("MaxPool"));
    EXPECT_TRUE(reg.has_op("BatchNormalization"));
    EXPECT_FALSE(reg.has_op("NonExistent"));
}

TEST(RegistryTest, DispatchRelu) {
    register_all_ops();
    auto& reg = OpRegistry::instance();

    OpContext ctx;
    ctx.inputs.push_back(Tensor({3}, {-1.0f, 0.0f, 2.0f}));

    Tensor out = reg.dispatch("Relu", ctx);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
}

TEST(RegistryTest, DispatchUnknownOp) {
    register_all_ops();
    auto& reg = OpRegistry::instance();

    OpContext ctx;
    EXPECT_THROW(reg.dispatch("FakeOp", ctx), std::runtime_error);
}

TEST(RegistryTest, ListOps) {
    register_all_ops();
    auto& reg = OpRegistry::instance();
    auto ops = reg.list_ops();
    EXPECT_GE(ops.size(), 8u);
}

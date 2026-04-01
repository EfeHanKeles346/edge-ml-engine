#include <gtest/gtest.h>
#include "edge_ml/tensor.h"

using namespace edge_ml;

TEST(TensorTest, CreateEmpty) {
    Tensor t;
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.ndim(), 0);
}

TEST(TensorTest, CreateWithShape) {
    Tensor t({2, 3});
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, CreateWithData) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[5], 6.0f);
}

TEST(TensorTest, DataSizeMismatch) {
    EXPECT_THROW(Tensor({2, 3}, {1, 2, 3}), std::invalid_argument);
}

TEST(TensorTest, Fill) {
    Tensor t({2, 3});
    t.fill(3.14f);
    for (int i = 0; i < t.size(); i++) {
        EXPECT_FLOAT_EQ(t[i], 3.14f);
    }
}

TEST(TensorTest, AtMultiIndex) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    // Row 0: [1, 2, 3]
    // Row 1: [4, 5, 6]
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(TensorTest, AtWriteAccess) {
    Tensor t({2, 2});
    t.at({0, 0}) = 10.0f;
    t.at({1, 1}) = 20.0f;
    EXPECT_FLOAT_EQ(t.at({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 20.0f);
}

TEST(TensorTest, CopyConstructor) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b(a);
    EXPECT_FLOAT_EQ(b[0], 1.0f);
    EXPECT_FLOAT_EQ(b[3], 4.0f);
    // Modify original, copy should be independent
    a[0] = 99.0f;
    EXPECT_FLOAT_EQ(b[0], 1.0f);
}

TEST(TensorTest, MoveConstructor) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b(std::move(a));
    EXPECT_FLOAT_EQ(b[0], 1.0f);
    EXPECT_FLOAT_EQ(b[3], 4.0f);
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
}

TEST(TensorTest, ThreeDimensional) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.ndim(), 3);
    t.at({1, 2, 3}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 42.0f);
}

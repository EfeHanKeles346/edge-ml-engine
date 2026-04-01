#include <gtest/gtest.h>
#include "edge_ml/tensor.h"

using namespace edge_ml;

// ============================================================
// Basic (Week 1)
// ============================================================

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
    // shared_ptr: both point to same data
    // this is expected with the new design
}

TEST(TensorTest, MoveConstructor) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b(std::move(a));
    EXPECT_FLOAT_EQ(b[0], 1.0f);
    EXPECT_FLOAT_EQ(b[3], 4.0f);
    EXPECT_EQ(a.size(), 0);
}

TEST(TensorTest, ThreeDimensional) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.ndim(), 3);
    t.at({1, 2, 3}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 42.0f);
}

// ============================================================
// Stride (Week 2)
// ============================================================

TEST(TensorTest, StridesRowMajor) {
    Tensor t({2, 3});
    // Row-major: stride = [3, 1]
    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);
}

TEST(TensorTest, Strides3D) {
    Tensor t({2, 3, 4});
    // [3*4, 4, 1] = [12, 4, 1]
    EXPECT_EQ(t.strides()[0], 12);
    EXPECT_EQ(t.strides()[1], 4);
    EXPECT_EQ(t.strides()[2], 1);
}

TEST(TensorTest, IsContiguous) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_TRUE(t.is_contiguous());
}

// ============================================================
// Reshape (Week 2)
// ============================================================

TEST(TensorTest, ReshapeBasic) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.reshape({3, 2});
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 2);
    // Data order preserved: [1,2,3,4,5,6]
    EXPECT_FLOAT_EQ(r.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(r.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(r.at({2, 1}), 6.0f);
}

TEST(TensorTest, ReshapeFlatten) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.reshape({6});
    EXPECT_EQ(r.ndim(), 1);
    EXPECT_EQ(r.shape()[0], 6);
    EXPECT_FLOAT_EQ(r.at({4}), 5.0f);
}

TEST(TensorTest, ReshapeInferDim) {
    Tensor t({2, 3, 4});
    Tensor r = t.reshape({6, -1});
    EXPECT_EQ(r.shape()[0], 6);
    EXPECT_EQ(r.shape()[1], 4);
}

TEST(TensorTest, ReshapeZeroCopy) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.reshape({3, 2});
    // Zero-copy: same underlying data
    EXPECT_EQ(t.data(), r.data());
}

TEST(TensorTest, ReshapeSizeMismatch) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument);
}

// ============================================================
// Transpose (Week 2)
// ============================================================

TEST(TensorTest, Transpose2D) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Tensor tr = t.transpose(0, 1);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    EXPECT_EQ(tr.shape()[0], 3);
    EXPECT_EQ(tr.shape()[1], 2);
    EXPECT_FLOAT_EQ(tr.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tr.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(tr.at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(tr.at({2, 1}), 6.0f);
}

TEST(TensorTest, TransposeZeroCopy) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tr = t.transpose(0, 1);
    // Zero-copy: same underlying data
    EXPECT_EQ(t.data(), tr.data());
}

TEST(TensorTest, TransposeNotContiguous) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tr = t.transpose(0, 1);
    EXPECT_FALSE(tr.is_contiguous());
}

TEST(TensorTest, TransposeMakeContiguous) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tr = t.transpose(0, 1).contiguous();
    EXPECT_TRUE(tr.is_contiguous());
    // Data should be: [1, 4, 2, 5, 3, 6]
    EXPECT_FLOAT_EQ(tr[0], 1.0f);
    EXPECT_FLOAT_EQ(tr[1], 4.0f);
    EXPECT_FLOAT_EQ(tr[2], 2.0f);
    EXPECT_FLOAT_EQ(tr[3], 5.0f);
}

TEST(TensorTest, TransposeInvalidDim) {
    Tensor t({2, 3});
    EXPECT_THROW(t.transpose(0, 5), std::invalid_argument);
}

// ============================================================
// Element-wise ops (Week 2)
// ============================================================

TEST(TensorTest, Add) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {10, 20, 30, 40});
    Tensor c = a.add(b);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 33.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 44.0f);
}

TEST(TensorTest, Subtract) {
    Tensor a({2, 2}, {10, 20, 30, 40});
    Tensor b({2, 2}, {1, 2, 3, 4});
    Tensor c = a.subtract(b);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 36.0f);
}

TEST(TensorTest, Multiply) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor c = a.multiply(b);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 12.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 21.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 32.0f);
}

TEST(TensorTest, AddShapeMismatch) {
    Tensor a({2, 2});
    Tensor b({3, 3});
    EXPECT_THROW(a.add(b), std::invalid_argument);
}

TEST(TensorTest, ElementWiseOnTransposed) {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {10, 20, 30, 40, 50, 60});
    // Transpose both to (3,2), then add
    Tensor at = a.transpose(0, 1);
    Tensor bt = b.transpose(0, 1);
    Tensor c = at.add(bt);
    // at = [[1,4],[2,5],[3,6]]  bt = [[10,40],[20,50],[30,60]]
    EXPECT_FLOAT_EQ(c.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 44.0f);
    EXPECT_FLOAT_EQ(c.at({2, 1}), 66.0f);
}

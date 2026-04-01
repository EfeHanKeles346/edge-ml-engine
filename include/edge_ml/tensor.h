#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstring>
#include <stdexcept>

namespace edge_ml {

class Tensor {
public:
    // Constructors
    Tensor();
    explicit Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    // Rule of five
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;

    // Element access (flat index — raw buffer)
    float& operator[](int index);
    const float& operator[](int index) const;

    // Element access (multi-dim — uses strides)
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    // Info
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int size() const { return size_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    float* data() { return data_.get() + offset_; }
    const float* data() const { return data_.get() + offset_; }
    bool is_contiguous() const;

    // Shape operations (zero-copy)
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor contiguous() const;

    // Element-wise operations (return new tensor)
    Tensor add(const Tensor& other) const;
    Tensor subtract(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const;
    Tensor matmul_tiled(const Tensor& other, int tile_size = 32) const;

    // Utilities
    void fill(float value);
    void print() const;

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    std::shared_ptr<float[]> data_;
    int size_;
    int offset_;

    void compute_strides();
    int flat_index(const std::vector<int>& indices) const;
    void increment_indices(std::vector<int>& indices) const;
};

} // namespace edge_ml

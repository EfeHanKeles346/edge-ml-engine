#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace edge_ml {

class Tensor {
public:
    // Constructors
    Tensor();
    explicit Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    // Copy & Move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor
    ~Tensor();

    // Element access
    float& operator[](int index);
    const float& operator[](int index) const;
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    // Info
    const std::vector<int>& shape() const { return shape_; }
    int size() const { return size_; }
    int ndim() const { return shape_.size(); }
    float* data() { return data_; }
    const float* data() const { return data_; }

    // Utilities
    void fill(float value);
    void print() const;

private:
    std::vector<int> shape_;
    float* data_;
    int size_;

    int compute_size() const;
    int flat_index(const std::vector<int>& indices) const;
};

} // namespace edge_ml

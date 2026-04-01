#include "edge_ml/tensor.h"

namespace edge_ml {

Tensor::Tensor() : data_(nullptr), size_(0) {}

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    size_ = compute_size();
    data_ = new float[size_]();
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape) {
    size_ = compute_size();
    if (static_cast<int>(data.size()) != size_) {
        throw std::invalid_argument("Data size does not match shape");
    }
    data_ = new float[size_];
    std::memcpy(data_, data.data(), size_ * sizeof(float));
}

// Copy constructor
Tensor::Tensor(const Tensor& other) : shape_(other.shape_), size_(other.size_) {
    data_ = new float[size_];
    std::memcpy(data_, other.data_, size_ * sizeof(float));
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data_;
        shape_ = other.shape_;
        size_ = other.size_;
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
    return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        shape_ = std::move(other.shape_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    delete[] data_;
}

float& Tensor::operator[](int index) {
    return data_[index];
}

const float& Tensor::operator[](int index) const {
    return data_[index];
}

float& Tensor::at(const std::vector<int>& indices) {
    return data_[flat_index(indices)];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    return data_[flat_index(indices)];
}

void Tensor::fill(float value) {
    for (int i = 0; i < size_; i++) {
        data_[i] = value;
    }
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (int i = 0; i < static_cast<int>(shape_.size()); i++) {
        std::cout << shape_[i];
        if (i < static_cast<int>(shape_.size()) - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    int limit = std::min(size_, 20);
    for (int i = 0; i < limit; i++) {
        std::cout << data_[i];
        if (i < limit - 1) std::cout << ", ";
    }
    if (size_ > 20) std::cout << ", ...";
    std::cout << "])" << std::endl;
}

int Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    int s = 1;
    for (int dim : shape_) s *= dim;
    return s;
}

int Tensor::flat_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimension mismatch");
    }
    int idx = 0;
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape_[i];
    }
    return idx;
}

} // namespace edge_ml

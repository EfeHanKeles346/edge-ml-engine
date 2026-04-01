#include "edge_ml/tensor.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace edge_ml {

// ============================================================
// Constructors
// ============================================================

Tensor::Tensor() : size_(0), offset_(0) {}

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape), offset_(0) {
    size_ = 1;
    for (int d : shape_) size_ *= d;
    compute_strides();
    data_ = std::shared_ptr<float[]>(new float[size_]());
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape), offset_(0) {
    size_ = 1;
    for (int d : shape_) size_ *= d;
    if (static_cast<int>(data.size()) != size_) {
        throw std::invalid_argument("Data size does not match shape");
    }
    compute_strides();
    data_ = std::shared_ptr<float[]>(new float[size_]);
    std::memcpy(data_.get(), data.data(), size_ * sizeof(float));
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      data_(std::move(other.data_)),
      size_(other.size_),
      offset_(other.offset_) {
    other.size_ = 0;
    other.offset_ = 0;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        data_ = std::move(other.data_);
        size_ = other.size_;
        offset_ = other.offset_;
        other.size_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

// ============================================================
// Element access
// ============================================================

float& Tensor::operator[](int index) {
    return data_.get()[offset_ + index];
}

const float& Tensor::operator[](int index) const {
    return data_.get()[offset_ + index];
}

float& Tensor::at(const std::vector<int>& indices) {
    return data_.get()[flat_index(indices)];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    return data_.get()[flat_index(indices)];
}

// ============================================================
// Info
// ============================================================

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    int expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

// ============================================================
// Shape operations
// ============================================================

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    if (!is_contiguous()) {
        // Must copy first
        return contiguous().reshape(new_shape);
    }

    // Handle -1 dimension (infer)
    std::vector<int> resolved = new_shape;
    int unknown_idx = -1;
    int known_product = 1;
    for (int i = 0; i < static_cast<int>(resolved.size()); i++) {
        if (resolved[i] == -1) {
            if (unknown_idx != -1) {
                throw std::invalid_argument("Only one -1 dimension allowed");
            }
            unknown_idx = i;
        } else {
            known_product *= resolved[i];
        }
    }
    if (unknown_idx != -1) {
        resolved[unknown_idx] = size_ / known_product;
    }

    // Verify total size matches
    int new_size = 1;
    for (int d : resolved) new_size *= d;
    if (new_size != size_) {
        throw std::invalid_argument("Cannot reshape: size mismatch");
    }

    Tensor result;
    result.shape_ = resolved;
    result.data_ = data_;  // shared — zero copy
    result.size_ = size_;
    result.offset_ = offset_;
    result.compute_strides();
    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim()) {
        throw std::invalid_argument("Transpose dimension out of range");
    }

    Tensor result;
    result.shape_ = shape_;
    result.strides_ = strides_;
    result.data_ = data_;  // shared — zero copy
    result.size_ = size_;
    result.offset_ = offset_;

    // Swap shape and stride for the two dims
    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.strides_[dim0], result.strides_[dim1]);
    return result;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;

    Tensor result(shape_);
    std::vector<int> indices(ndim(), 0);
    for (int i = 0; i < size_; i++) {
        result.data_.get()[i] = at(indices);
        increment_indices(indices);
    }
    return result;
}

// ============================================================
// Element-wise operations
// ============================================================

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for add");
    }
    Tensor result(shape_);
    std::vector<int> indices(ndim(), 0);
    for (int i = 0; i < size_; i++) {
        result.data_.get()[i] = at(indices) + other.at(indices);
        increment_indices(indices);
    }
    return result;
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for subtract");
    }
    Tensor result(shape_);
    std::vector<int> indices(ndim(), 0);
    for (int i = 0; i < size_; i++) {
        result.data_.get()[i] = at(indices) - other.at(indices);
        increment_indices(indices);
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for multiply");
    }
    Tensor result(shape_);
    std::vector<int> indices(ndim(), 0);
    for (int i = 0; i < size_; i++) {
        result.data_.get()[i] = at(indices) * other.at(indices);
        increment_indices(indices);
    }
    return result;
}

// ============================================================
// Matrix multiplication
// ============================================================

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    int M = shape_[0];   // rows of A
    int K = shape_[1];   // cols of A = rows of B
    int N = other.shape_[1]; // cols of B

    if (K != other.shape_[0]) {
        throw std::invalid_argument("matmul shape mismatch: A cols != B rows");
    }

    // Make contiguous copies if needed
    Tensor a = is_contiguous() ? *this : contiguous();
    Tensor b = other.is_contiguous() ? other : other.contiguous();
    const float* A = a.data();
    const float* B = b.data();

    Tensor result({M, N});
    float* C = result.data();

    // Naive: 3 nested loops (i-j-k)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return result;
}

Tensor Tensor::matmul_tiled(const Tensor& other, int tile_size) const {
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("matmul_tiled requires 2D tensors");
    }
    int M = shape_[0];
    int K = shape_[1];
    int N = other.shape_[1];

    if (K != other.shape_[0]) {
        throw std::invalid_argument("matmul_tiled shape mismatch: A cols != B rows");
    }

    Tensor a = is_contiguous() ? *this : contiguous();
    Tensor b = other.is_contiguous() ? other : other.contiguous();
    const float* A = a.data();
    const float* B = b.data();

    Tensor result({M, N});
    result.fill(0.0f);
    float* C = result.data();

    // Tiled: i-j-k with blocks for better cache usage
    for (int i0 = 0; i0 < M; i0 += tile_size) {
        int imax = std::min(i0 + tile_size, M);
        for (int k0 = 0; k0 < K; k0 += tile_size) {
            int kmax = std::min(k0 + tile_size, K);
            for (int j0 = 0; j0 < N; j0 += tile_size) {
                int jmax = std::min(j0 + tile_size, N);
                // Multiply tile
                for (int i = i0; i < imax; i++) {
                    for (int k = k0; k < kmax; k++) {
                        float a_val = A[i * K + k];
                        for (int j = j0; j < jmax; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
    return result;
}

// ============================================================
// Utilities
// ============================================================

void Tensor::fill(float value) {
    if (is_contiguous()) {
        for (int i = 0; i < size_; i++) {
            data_.get()[offset_ + i] = value;
        }
    } else {
        std::vector<int> indices(ndim(), 0);
        for (int i = 0; i < size_; i++) {
            data_.get()[flat_index(indices)] = value;
            increment_indices(indices);
        }
    }
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (int i = 0; i < ndim(); i++) {
        std::cout << shape_[i];
        if (i < ndim() - 1) std::cout << ", ";
    }
    std::cout << "], stride=[";
    for (int i = 0; i < ndim(); i++) {
        std::cout << strides_[i];
        if (i < ndim() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";

    std::vector<int> indices(ndim(), 0);
    int limit = std::min(size_, 20);
    for (int i = 0; i < limit; i++) {
        std::cout << at(indices);
        if (i < limit - 1) std::cout << ", ";
        increment_indices(indices);
    }
    if (size_ > 20) std::cout << ", ...";
    std::cout << "])" << std::endl;
}

// ============================================================
// Private helpers
// ============================================================

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

int Tensor::flat_index(const std::vector<int>& indices) const {
    if (static_cast<int>(indices.size()) != ndim()) {
        throw std::invalid_argument("Index dimension mismatch");
    }
    int idx = offset_;
    for (int i = 0; i < ndim(); i++) {
        idx += indices[i] * strides_[i];
    }
    return idx;
}

void Tensor::increment_indices(std::vector<int>& indices) const {
    for (int i = ndim() - 1; i >= 0; i--) {
        indices[i]++;
        if (indices[i] < shape_[i]) return;
        indices[i] = 0;
    }
}

} // namespace edge_ml

#include "edge_ml/graph/executor.h"
#include "edge_ml/operators/registry.h"
#include "edge_ml/operators/activations.h"
#include "edge_ml/operators/dense.h"
#include "edge_ml/operators/conv2d.h"
#include "edge_ml/operators/pooling.h"
#include "edge_ml/operators/batchnorm.h"
#include <stdexcept>
#include <iostream>

namespace edge_ml {

Executor::Executor(const OnnxModel& model) : model_(model) {
    ops::register_all_ops();
    graph_.build(model_);
    execution_order_ = graph_.topological_sort();
}

Tensor Executor::run(const Tensor& input) {
    if (model_.input_names.empty()) {
        throw std::runtime_error("Executor: model has no inputs");
    }
    std::unordered_map<std::string, Tensor> inputs;
    inputs[model_.input_names[0]] = input;
    return run(inputs);
}

Tensor Executor::run(const std::unordered_map<std::string, Tensor>& inputs) {
    // Tensor map: holds all intermediate tensors
    std::unordered_map<std::string, Tensor> tensor_map;

    // Load initializers (weights)
    for (const auto& pair : model_.initializers) {
        tensor_map[pair.first] = pair.second;
    }

    // Load inputs
    for (const auto& pair : inputs) {
        tensor_map[pair.first] = pair.second;
    }

    // Execute nodes in topological order
    for (int node_id : execution_order_) {
        const auto& node = graph_.node(node_id).onnx_node;
        Tensor result = execute_node(node, tensor_map);

        // Store output
        if (!node.outputs.empty()) {
            tensor_map[node.outputs[0]] = result;
        }
    }

    // Return the model output
    if (model_.output_names.empty()) {
        throw std::runtime_error("Executor: model has no outputs");
    }

    auto it = tensor_map.find(model_.output_names[0]);
    if (it == tensor_map.end()) {
        throw std::runtime_error("Executor: output tensor not found");
    }
    return it->second;
}

static int get_onnx_int_attr(const OnnxNode& node, const std::string& name, int default_val) {
    auto* attr = node.get_attr(name);
    if (attr) return attr->i;
    return default_val;
}

static std::vector<int> get_onnx_ints_attr(const OnnxNode& node, const std::string& name) {
    auto* attr = node.get_attr(name);
    if (attr && !attr->ints.empty()) return attr->ints;
    return {};
}

static float get_onnx_float_attr(const OnnxNode& node, const std::string& name, float default_val) {
    auto* attr = node.get_attr(name);
    if (attr) return attr->f;
    return default_val;
}

Tensor Executor::execute_node(const OnnxNode& node,
                               std::unordered_map<std::string, Tensor>& tensor_map) {
    // Gather input tensors
    auto get_tensor = [&](int idx) -> const Tensor& {
        if (idx >= static_cast<int>(node.inputs.size()) || node.inputs[idx].empty()) {
            throw std::runtime_error("Executor: missing input " + std::to_string(idx) +
                                     " for op " + node.op_type);
        }
        auto it = tensor_map.find(node.inputs[idx]);
        if (it == tensor_map.end()) {
            throw std::runtime_error("Executor: tensor '" + node.inputs[idx] +
                                     "' not found for op " + node.op_type);
        }
        return it->second;
    };

    auto has_input = [&](int idx) -> bool {
        return idx < static_cast<int>(node.inputs.size()) &&
               !node.inputs[idx].empty() &&
               tensor_map.find(node.inputs[idx]) != tensor_map.end();
    };

    if (node.op_type == "Relu") {
        return ops::relu(get_tensor(0));
    }
    else if (node.op_type == "Sigmoid") {
        return ops::sigmoid(get_tensor(0));
    }
    else if (node.op_type == "Silu" || node.op_type == "SiLU") {
        return ops::silu(get_tensor(0));
    }
    else if (node.op_type == "Softmax") {
        int axis = get_onnx_int_attr(node, "axis", -1);
        return ops::softmax(get_tensor(0), axis);
    }
    else if (node.op_type == "MatMul") {
        const Tensor& a = get_tensor(0);
        const Tensor& b = get_tensor(1);
        return a.matmul(b);
    }
    else if (node.op_type == "Gemm") {
        // General Matrix Multiply: Y = alpha * A @ B + beta * C
        const Tensor& a = get_tensor(0);
        const Tensor& b = get_tensor(1);
        int transA = get_onnx_int_attr(node, "transA", 0);
        int transB = get_onnx_int_attr(node, "transB", 0);

        Tensor A = transA ? a.transpose(0, 1).contiguous() : a;
        Tensor B = transB ? b.transpose(0, 1).contiguous() : b;

        Tensor result = A.matmul(B);

        // Add bias if present
        if (has_input(2)) {
            const Tensor& c = get_tensor(2);
            float* dst = result.data();
            const float* bias = c.data();
            int rows = result.shape()[0];
            int cols = result.shape()[1];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    dst[i * cols + j] += bias[j];
                }
            }
        }
        return result;
    }
    else if (node.op_type == "Conv") {
        const Tensor& input = get_tensor(0);
        const Tensor& weight = get_tensor(1);

        // Get or create zero bias
        Tensor bias_tensor({weight.shape()[0]});
        bias_tensor.fill(0.0f);
        if (has_input(2)) {
            bias_tensor = get_tensor(2);
        }

        auto strides = get_onnx_ints_attr(node, "strides");
        auto pads = get_onnx_ints_attr(node, "pads");

        int sh = strides.size() >= 1 ? strides[0] : 1;
        int sw = strides.size() >= 2 ? strides[1] : sh;
        int ph = pads.size() >= 1 ? pads[0] : 0;
        int pw = pads.size() >= 2 ? pads[1] : ph;

        return ops::conv2d_im2col(input, weight, bias_tensor, sh, sw, ph, pw);
    }
    else if (node.op_type == "MaxPool") {
        const Tensor& input = get_tensor(0);

        auto kernel_shape = get_onnx_ints_attr(node, "kernel_shape");
        auto strides = get_onnx_ints_attr(node, "strides");
        auto pads = get_onnx_ints_attr(node, "pads");

        int kh = kernel_shape.size() >= 1 ? kernel_shape[0] : 2;
        int kw = kernel_shape.size() >= 2 ? kernel_shape[1] : kh;
        int sh = strides.size() >= 1 ? strides[0] : kh;
        int sw = strides.size() >= 2 ? strides[1] : kw;
        int ph = pads.size() >= 1 ? pads[0] : 0;
        int pw = pads.size() >= 2 ? pads[1] : ph;

        return ops::maxpool2d(input, kh, kw, sh, sw, ph, pw);
    }
    else if (node.op_type == "BatchNormalization") {
        const Tensor& input = get_tensor(0);
        const Tensor& scale = get_tensor(1);
        const Tensor& bias = get_tensor(2);
        const Tensor& mean = get_tensor(3);
        const Tensor& var = get_tensor(4);
        float eps = get_onnx_float_attr(node, "epsilon", 1e-5f);

        return ops::batchnorm(input, scale, bias, mean, var, eps);
    }
    else if (node.op_type == "Add") {
        return get_tensor(0).add(get_tensor(1));
    }
    else if (node.op_type == "Mul") {
        return get_tensor(0).multiply(get_tensor(1));
    }
    else if (node.op_type == "Sub") {
        return get_tensor(0).subtract(get_tensor(1));
    }
    else if (node.op_type == "Reshape") {
        const Tensor& input = get_tensor(0);
        const Tensor& shape_tensor = get_tensor(1);
        std::vector<int> new_shape;
        for (int i = 0; i < shape_tensor.size(); i++) {
            new_shape.push_back(static_cast<int>(shape_tensor[i]));
        }
        return input.contiguous().reshape(new_shape);
    }
    else if (node.op_type == "Flatten") {
        const Tensor& input = get_tensor(0);
        int axis = get_onnx_int_attr(node, "axis", 1);
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= input.shape()[i];
        for (int i = axis; i < input.ndim(); i++) inner *= input.shape()[i];
        return input.contiguous().reshape({outer, inner});
    }
    else if (node.op_type == "Transpose") {
        const Tensor& input = get_tensor(0);
        auto perm = get_onnx_ints_attr(node, "perm");
        if (perm.size() == 2 || (perm.size() == input.ndim())) {
            // For simple 2D transpose
            Tensor result = input;
            // Apply permutations as sequence of dim swaps
            // Simple case: just swap two dims
            if (input.ndim() == 2) {
                return input.transpose(0, 1);
            }
        }
        // For now, handle 2D transpose
        return input.transpose(0, 1);
    }
    else if (node.op_type == "GlobalAveragePool") {
        const Tensor& input = get_tensor(0);
        // Average over H and W dimensions (NCHW)
        int batch = input.shape()[0];
        int channels = input.shape()[1];
        int H = input.shape()[2];
        int W = input.shape()[3];
        int spatial = H * W;

        Tensor output({batch, channels, 1, 1});
        Tensor in = input.is_contiguous() ? input : input.contiguous();
        const float* in_data = in.data();
        float* out_data = output.data();

        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                int offset = (n * channels + c) * spatial;
                for (int s = 0; s < spatial; s++) {
                    sum += in_data[offset + s];
                }
                out_data[n * channels + c] = sum / spatial;
            }
        }
        return output;
    }
    else {
        throw std::runtime_error("Executor: unsupported op_type '" + node.op_type + "'");
    }
}

} // namespace edge_ml

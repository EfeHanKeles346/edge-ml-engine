#pragma once

#include "edge_ml/tensor.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace edge_ml {

struct OnnxAttribute {
    std::string name;
    int type; // 1=float, 2=int, 3=string, 6=floats, 7=ints
    float f;
    int i;
    std::string s;
    std::vector<float> floats;
    std::vector<int> ints;
};

struct OnnxNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<OnnxAttribute> attributes;

    // Helper to get attribute by name
    const OnnxAttribute* get_attr(const std::string& attr_name) const;
};

struct OnnxModel {
    std::vector<OnnxNode> nodes;
    std::unordered_map<std::string, Tensor> initializers; // weights
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    void print_summary() const;
};

OnnxModel parse_onnx(const std::string& filepath);

} // namespace edge_ml

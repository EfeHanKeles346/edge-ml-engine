#pragma once

#include "edge_ml/graph/graph.h"
#include "edge_ml/parser/onnx_parser.h"
#include "edge_ml/tensor.h"
#include <string>
#include <unordered_map>

namespace edge_ml {

class Executor {
public:
    Executor(const OnnxModel& model);

    Tensor run(const std::unordered_map<std::string, Tensor>& inputs);
    Tensor run(const Tensor& input);

private:
    OnnxModel model_;
    Graph graph_;
    std::vector<int> execution_order_;

    Tensor execute_node(const OnnxNode& node,
                        std::unordered_map<std::string, Tensor>& tensor_map);
};

} // namespace edge_ml

#pragma once

#include "edge_ml/parser/onnx_parser.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace edge_ml {

struct GraphNode {
    int id;
    OnnxNode onnx_node;
    std::vector<int> predecessors;
    std::vector<int> successors;
};

class Graph {
public:
    void build(const OnnxModel& model);
    std::vector<int> topological_sort() const;
    const GraphNode& node(int id) const { return nodes_[id]; }
    int num_nodes() const { return static_cast<int>(nodes_.size()); }
    void print() const;

private:
    std::vector<GraphNode> nodes_;
    // Maps tensor name -> node id that produces it
    std::unordered_map<std::string, int> producer_;
};

} // namespace edge_ml

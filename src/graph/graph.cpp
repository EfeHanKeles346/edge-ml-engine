#include "edge_ml/graph/graph.h"
#include <queue>
#include <stdexcept>
#include <iostream>

namespace edge_ml {

void Graph::build(const OnnxModel& model) {
    nodes_.clear();
    producer_.clear();

    // Create graph nodes
    for (int i = 0; i < static_cast<int>(model.nodes.size()); i++) {
        GraphNode gn;
        gn.id = i;
        gn.onnx_node = model.nodes[i];
        nodes_.push_back(gn);

        // Register this node as producer of its outputs
        for (const auto& output : model.nodes[i].outputs) {
            producer_[output] = i;
        }
    }

    // Build edges: for each node, check if its inputs are produced by another node
    for (int i = 0; i < static_cast<int>(nodes_.size()); i++) {
        for (const auto& input : nodes_[i].onnx_node.inputs) {
            if (input.empty()) continue;
            auto it = producer_.find(input);
            if (it != producer_.end()) {
                int pred = it->second;
                nodes_[i].predecessors.push_back(pred);
                nodes_[pred].successors.push_back(i);
            }
            // If not found in producer_, it's either an initializer or model input
        }
    }
}

std::vector<int> Graph::topological_sort() const {
    int n = static_cast<int>(nodes_.size());
    std::vector<int> in_degree(n, 0);

    for (int i = 0; i < n; i++) {
        in_degree[i] = static_cast<int>(nodes_[i].predecessors.size());
    }

    // Kahn's algorithm
    std::queue<int> q;
    for (int i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<int> order;
    order.reserve(n);

    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        order.push_back(curr);

        for (int succ : nodes_[curr].successors) {
            in_degree[succ]--;
            if (in_degree[succ] == 0) {
                q.push(succ);
            }
        }
    }

    if (static_cast<int>(order.size()) != n) {
        throw std::runtime_error("Graph has a cycle - topological sort failed");
    }

    return order;
}

void Graph::print() const {
    std::cout << "=== Computation Graph ===" << std::endl;
    for (const auto& node : nodes_) {
        std::cout << "Node " << node.id << ": " << node.onnx_node.op_type;
        if (!node.predecessors.empty()) {
            std::cout << " <- [";
            for (size_t i = 0; i < node.predecessors.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << node.predecessors[i];
            }
            std::cout << "]";
        }
        if (!node.successors.empty()) {
            std::cout << " -> [";
            for (size_t i = 0; i < node.successors.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << node.successors[i];
            }
            std::cout << "]";
        }
        std::cout << std::endl;
    }
}

} // namespace edge_ml

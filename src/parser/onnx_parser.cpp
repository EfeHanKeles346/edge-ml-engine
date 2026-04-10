#include "edge_ml/parser/onnx_parser.h"
#include "onnx.proto3.pb.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace edge_ml {

const OnnxAttribute* OnnxNode::get_attr(const std::string& attr_name) const {
    for (const auto& attr : attributes) {
        if (attr.name == attr_name) return &attr;
    }
    return nullptr;
}

void OnnxModel::print_summary() const {
    std::cout << "=== ONNX Model Summary ===" << std::endl;
    std::cout << "Inputs: ";
    for (const auto& name : input_names) std::cout << name << " ";
    std::cout << std::endl;
    std::cout << "Outputs: ";
    for (const auto& name : output_names) std::cout << name << " ";
    std::cout << std::endl;
    std::cout << "Initializers (weights): " << initializers.size() << std::endl;
    std::cout << "Nodes: " << nodes.size() << std::endl;
    for (size_t i = 0; i < nodes.size(); i++) {
        const auto& node = nodes[i];
        std::cout << "  [" << i << "] " << node.op_type;
        if (!node.name.empty()) std::cout << " (" << node.name << ")";
        std::cout << " | in: ";
        for (const auto& in : node.inputs) std::cout << in << " ";
        std::cout << "| out: ";
        for (const auto& out : node.outputs) std::cout << out << " ";
        std::cout << std::endl;
    }
}

static Tensor tensor_from_proto(const onnx::TensorProto& tp) {
    std::vector<int> shape;
    for (int d = 0; d < tp.dims_size(); d++) {
        shape.push_back(static_cast<int>(tp.dims(d)));
    }

    if (shape.empty()) {
        shape.push_back(1);
    }

    Tensor t(shape);
    float* data = t.data();

    if (tp.data_type() == onnx::TensorProto::FLOAT) {
        if (tp.float_data_size() > 0) {
            for (int i = 0; i < tp.float_data_size() && i < t.size(); i++) {
                data[i] = tp.float_data(i);
            }
        } else if (!tp.raw_data().empty()) {
            const float* raw = reinterpret_cast<const float*>(tp.raw_data().data());
            int count = static_cast<int>(tp.raw_data().size() / sizeof(float));
            for (int i = 0; i < count && i < t.size(); i++) {
                data[i] = raw[i];
            }
        }
    } else if (tp.data_type() == onnx::TensorProto::DOUBLE) {
        if (tp.double_data_size() > 0) {
            for (int i = 0; i < tp.double_data_size() && i < t.size(); i++) {
                data[i] = static_cast<float>(tp.double_data(i));
            }
        } else if (!tp.raw_data().empty()) {
            const double* raw = reinterpret_cast<const double*>(tp.raw_data().data());
            int count = static_cast<int>(tp.raw_data().size() / sizeof(double));
            for (int i = 0; i < count && i < t.size(); i++) {
                data[i] = static_cast<float>(raw[i]);
            }
        }
    } else if (tp.data_type() == onnx::TensorProto::INT64) {
        if (tp.int64_data_size() > 0) {
            for (int i = 0; i < tp.int64_data_size() && i < t.size(); i++) {
                data[i] = static_cast<float>(tp.int64_data(i));
            }
        } else if (!tp.raw_data().empty()) {
            const int64_t* raw = reinterpret_cast<const int64_t*>(tp.raw_data().data());
            int count = static_cast<int>(tp.raw_data().size() / sizeof(int64_t));
            for (int i = 0; i < count && i < t.size(); i++) {
                data[i] = static_cast<float>(raw[i]);
            }
        }
    }

    return t;
}

OnnxModel parse_onnx(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ONNX file: " + filepath);
    }

    onnx::ModelProto model_proto;
    if (!model_proto.ParseFromIstream(&file)) {
        throw std::runtime_error("Failed to parse ONNX file: " + filepath);
    }

    OnnxModel model;
    const auto& graph = model_proto.graph();

    // Parse initializers (weights)
    for (int i = 0; i < graph.initializer_size(); i++) {
        const auto& init = graph.initializer(i);
        model.initializers[init.name()] = tensor_from_proto(init);
    }

    // Parse inputs (exclude initializers - they are weights, not real inputs)
    for (int i = 0; i < graph.input_size(); i++) {
        const std::string& name = graph.input(i).name();
        if (model.initializers.find(name) == model.initializers.end()) {
            model.input_names.push_back(name);
        }
    }

    // Parse outputs
    for (int i = 0; i < graph.output_size(); i++) {
        model.output_names.push_back(graph.output(i).name());
    }

    // Parse nodes
    for (int i = 0; i < graph.node_size(); i++) {
        const auto& proto_node = graph.node(i);
        OnnxNode node;
        node.op_type = proto_node.op_type();
        node.name = proto_node.name();

        for (int j = 0; j < proto_node.input_size(); j++) {
            node.inputs.push_back(proto_node.input(j));
        }
        for (int j = 0; j < proto_node.output_size(); j++) {
            node.outputs.push_back(proto_node.output(j));
        }

        // Parse attributes
        for (int j = 0; j < proto_node.attribute_size(); j++) {
            const auto& proto_attr = proto_node.attribute(j);
            OnnxAttribute attr;
            attr.name = proto_attr.name();
            attr.type = proto_attr.type();
            attr.f = proto_attr.f();
            attr.i = static_cast<int>(proto_attr.i());
            attr.s = proto_attr.s();
            for (int k = 0; k < proto_attr.floats_size(); k++) {
                attr.floats.push_back(proto_attr.floats(k));
            }
            for (int k = 0; k < proto_attr.ints_size(); k++) {
                attr.ints.push_back(static_cast<int>(proto_attr.ints(k)));
            }
            node.attributes.push_back(attr);
        }

        model.nodes.push_back(node);
    }

    return model;
}

} // namespace edge_ml

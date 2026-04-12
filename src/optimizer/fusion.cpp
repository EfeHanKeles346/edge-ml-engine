#include "edge_ml/optimizer/fusion.h"
#include <cmath>
#include <iostream>

namespace edge_ml {
namespace opt {

void fuse_conv_batchnorm(Tensor& conv_weight, Tensor& conv_bias,
                         const Tensor& bn_gamma, const Tensor& bn_beta,
                         const Tensor& bn_mean, const Tensor& bn_var,
                         float epsilon) {
    // Fuse BatchNorm into Conv weights:
    // new_weight[oc] = weight[oc] * gamma[oc] / sqrt(var[oc] + eps)
    // new_bias[oc] = (bias[oc] - mean[oc]) * gamma[oc] / sqrt(var[oc] + eps) + beta[oc]

    int out_channels = conv_weight.shape()[0];
    int channels_per_filter = conv_weight.size() / out_channels;

    float* w_data = conv_weight.data();
    float* b_data = conv_bias.data();
    const float* gamma = bn_gamma.data();
    const float* beta = bn_beta.data();
    const float* mean = bn_mean.data();
    const float* var = bn_var.data();

    for (int oc = 0; oc < out_channels; oc++) {
        float scale = gamma[oc] / std::sqrt(var[oc] + epsilon);

        // Scale all weights for this output channel
        for (int i = 0; i < channels_per_filter; i++) {
            w_data[oc * channels_per_filter + i] *= scale;
        }

        // Update bias
        b_data[oc] = (b_data[oc] - mean[oc]) * scale + beta[oc];
    }
}

int fuse_graph(OnnxModel& model) {
    int fusions = 0;

    // Pass 1: Conv + BatchNorm fusion
    for (size_t i = 0; i + 1 < model.nodes.size(); i++) {
        auto& conv_node = model.nodes[i];
        auto& bn_node = model.nodes[i + 1];

        if (conv_node.op_type != "Conv" || bn_node.op_type != "BatchNormalization") {
            continue;
        }

        // Check if Conv output feeds into BatchNorm input
        if (conv_node.outputs.empty() || bn_node.inputs.empty() ||
            conv_node.outputs[0] != bn_node.inputs[0]) {
            continue;
        }

        // Get Conv weights
        std::string conv_w_name = conv_node.inputs[1];
        std::string conv_b_name = conv_node.inputs.size() > 2 ? conv_node.inputs[2] : "";

        // Get BN parameters
        if (bn_node.inputs.size() < 5) continue;
        std::string bn_gamma_name = bn_node.inputs[1];
        std::string bn_beta_name = bn_node.inputs[2];
        std::string bn_mean_name = bn_node.inputs[3];
        std::string bn_var_name = bn_node.inputs[4];

        // Check all weights exist
        auto& inits = model.initializers;
        if (!inits.count(conv_w_name) || !inits.count(bn_gamma_name) ||
            !inits.count(bn_beta_name) || !inits.count(bn_mean_name) ||
            !inits.count(bn_var_name)) {
            continue;
        }

        // Create conv bias if it doesn't exist
        if (conv_b_name.empty() || !inits.count(conv_b_name)) {
            conv_b_name = conv_w_name + "_bias_fused";
            int out_channels = inits[conv_w_name].shape()[0];
            Tensor bias({out_channels});
            bias.fill(0.0f);
            inits[conv_b_name] = bias;
            if (conv_node.inputs.size() < 3) {
                conv_node.inputs.push_back(conv_b_name);
            } else {
                conv_node.inputs[2] = conv_b_name;
            }
        }

        // Get BN epsilon
        float eps = 1e-5f;
        auto* eps_attr = bn_node.get_attr("epsilon");
        if (eps_attr) eps = eps_attr->f;

        // Fuse!
        fuse_conv_batchnorm(inits[conv_w_name], inits[conv_b_name],
                           inits[bn_gamma_name], inits[bn_beta_name],
                           inits[bn_mean_name], inits[bn_var_name], eps);

        // Redirect Conv output to BN output
        conv_node.outputs[0] = bn_node.outputs[0];

        // Mark BN node for removal (set op_type to empty)
        bn_node.op_type = "";
        fusions++;
    }

    // Pass 2: Conv + ReLU fusion (mark for executor optimization)
    for (size_t i = 0; i + 1 < model.nodes.size(); i++) {
        auto& conv_node = model.nodes[i];
        auto& relu_node = model.nodes[i + 1];

        if (conv_node.op_type != "Conv" || relu_node.op_type != "Relu") {
            continue;
        }

        if (conv_node.outputs.empty() || relu_node.inputs.empty() ||
            conv_node.outputs[0] != relu_node.inputs[0]) {
            continue;
        }

        // Redirect Conv output to ReLU output
        conv_node.outputs[0] = relu_node.outputs[0];

        // Add a fused_relu attribute to Conv
        OnnxAttribute attr;
        attr.name = "fused_relu";
        attr.i = 1;
        attr.type = 2;
        conv_node.attributes.push_back(attr);

        // Mark ReLU for removal
        relu_node.op_type = "";
        fusions++;
    }

    // Remove empty nodes
    std::vector<OnnxNode> cleaned;
    for (auto& node : model.nodes) {
        if (!node.op_type.empty()) {
            cleaned.push_back(std::move(node));
        }
    }
    model.nodes = std::move(cleaned);

    return fusions;
}

} // namespace opt
} // namespace edge_ml

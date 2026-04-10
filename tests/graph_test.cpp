#include <gtest/gtest.h>
#include "edge_ml/parser/onnx_parser.h"
#include "edge_ml/graph/graph.h"
#include "edge_ml/graph/executor.h"
#include <cmath>

using namespace edge_ml;

static const std::string MODEL_PATH = std::string(TEST_DATA_DIR) + "/test_model.onnx";

// ==================== ONNX Parser Tests ====================

TEST(OnnxParserTest, ParseTestModel) {
    OnnxModel model = parse_onnx(MODEL_PATH);

    // Should have nodes
    EXPECT_EQ(model.nodes.size(), 6u); // Conv, Relu, MaxPool, Flatten, Gemm, Softmax

    // Should have weights
    EXPECT_TRUE(model.initializers.count("conv_w"));
    EXPECT_TRUE(model.initializers.count("conv_b"));
    EXPECT_TRUE(model.initializers.count("fc_w"));
    EXPECT_TRUE(model.initializers.count("fc_b"));

    // Check weight shapes
    EXPECT_EQ(model.initializers["conv_w"].shape(), (std::vector<int>{2, 1, 3, 3}));
    EXPECT_EQ(model.initializers["fc_w"].shape(), (std::vector<int>{3, 18}));

    // Should have 1 input (not counting initializers)
    EXPECT_EQ(model.input_names.size(), 1u);
    EXPECT_EQ(model.input_names[0], "input");

    // Should have 1 output
    EXPECT_EQ(model.output_names.size(), 1u);
    EXPECT_EQ(model.output_names[0], "output");
}

TEST(OnnxParserTest, NodeOpTypes) {
    OnnxModel model = parse_onnx(MODEL_PATH);

    std::vector<std::string> expected_ops = {"Conv", "Relu", "MaxPool", "Flatten", "Gemm", "Softmax"};
    ASSERT_EQ(model.nodes.size(), expected_ops.size());
    for (size_t i = 0; i < expected_ops.size(); i++) {
        EXPECT_EQ(model.nodes[i].op_type, expected_ops[i]);
    }
}

TEST(OnnxParserTest, ConvAttributes) {
    OnnxModel model = parse_onnx(MODEL_PATH);
    const auto& conv = model.nodes[0];
    EXPECT_EQ(conv.op_type, "Conv");

    auto* kernel_shape = conv.get_attr("kernel_shape");
    ASSERT_NE(kernel_shape, nullptr);
    EXPECT_EQ(kernel_shape->ints, (std::vector<int>{3, 3}));
}

TEST(OnnxParserTest, InvalidFile) {
    EXPECT_THROW(parse_onnx("nonexistent.onnx"), std::runtime_error);
}

// ==================== Graph Tests ====================

TEST(GraphTest, BuildFromModel) {
    OnnxModel model = parse_onnx(MODEL_PATH);
    Graph graph;
    graph.build(model);

    EXPECT_EQ(graph.num_nodes(), 6);
}

TEST(GraphTest, TopologicalSort) {
    OnnxModel model = parse_onnx(MODEL_PATH);
    Graph graph;
    graph.build(model);

    auto order = graph.topological_sort();
    EXPECT_EQ(static_cast<int>(order.size()), graph.num_nodes());

    // Verify: each node appears after its predecessors
    std::unordered_set<int> visited;
    for (int id : order) {
        for (int pred : graph.node(id).predecessors) {
            EXPECT_TRUE(visited.count(pred)) << "Node " << id << " before predecessor " << pred;
        }
        visited.insert(id);
    }
}

TEST(GraphTest, LinearChain) {
    // The test model is a linear chain: Conv -> Relu -> MaxPool -> Flatten -> Gemm -> Softmax
    OnnxModel model = parse_onnx(MODEL_PATH);
    Graph graph;
    graph.build(model);

    auto order = graph.topological_sort();
    // Should be in order 0,1,2,3,4,5
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(order[i], i);
    }
}

// ==================== Executor Tests ====================

TEST(ExecutorTest, RunTestModel) {
    OnnxModel model = parse_onnx(MODEL_PATH);
    Executor executor(model);

    // Input: all ones, shape {1, 1, 8, 8}
    Tensor input({1, 1, 8, 8});
    input.fill(1.0f);

    Tensor output = executor.run(input);

    // Output should be shape {1, 3}
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);

    // Softmax output should sum to ~1.0
    float sum = 0;
    for (int i = 0; i < output.size(); i++) {
        sum += output[i];
        EXPECT_GE(output[i], 0.0f); // all probabilities >= 0
        EXPECT_LE(output[i], 1.0f); // all probabilities <= 1
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4);

    // Validate against PyTorch/ONNXRuntime expected output
    EXPECT_NEAR(output.at({0, 0}), 0.01313143f, 1e-3);
    EXPECT_NEAR(output.at({0, 1}), 0.01528960f, 1e-3);
    EXPECT_NEAR(output.at({0, 2}), 0.97157896f, 1e-3);
}

TEST(ExecutorTest, RunWithMap) {
    OnnxModel model = parse_onnx(MODEL_PATH);
    Executor executor(model);

    Tensor input({1, 1, 8, 8});
    input.fill(1.0f);

    std::unordered_map<std::string, Tensor> inputs;
    inputs["input"] = input;

    Tensor output = executor.run(inputs);
    EXPECT_EQ(output.size(), 3);

    float sum = 0;
    for (int i = 0; i < output.size(); i++) sum += output[i];
    EXPECT_NEAR(sum, 1.0f, 1e-4);
}

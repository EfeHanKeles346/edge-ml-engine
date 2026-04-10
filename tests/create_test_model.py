"""
Create a simple ONNX test model: Conv2D -> ReLU -> MaxPool -> Flatten -> Gemm -> Softmax
Input: {1, 1, 8, 8}  Output: {1, 3}
"""
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("pip install onnx numpy")
    exit(1)

np.random.seed(42)

# Conv2D weights: {2, 1, 3, 3}
conv_w = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.5
conv_b = np.zeros(2, dtype=np.float32)

# FC weights: {3, 18} (after maxpool: 2 channels * 3 * 3 = 18)
fc_w = np.random.randn(3, 18).astype(np.float32) * 0.5
fc_b = np.zeros(3, dtype=np.float32)

# Create nodes
conv_node = helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv_out"],
                              kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"])
pool_node = helper.make_node("MaxPool", ["relu_out"], ["pool_out"],
                              kernel_shape=[2, 2], strides=[2, 2])
flatten_node = helper.make_node("Flatten", ["pool_out"], ["flat_out"], axis=1)
gemm_node = helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["gemm_out"],
                              transB=1)
softmax_node = helper.make_node("Softmax", ["gemm_out"], ["output"], axis=1)

# Create graph
graph = helper.make_graph(
    [conv_node, relu_node, pool_node, flatten_node, gemm_node, softmax_node],
    "test_cnn",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    initializer=[
        numpy_helper.from_array(conv_w, "conv_w"),
        numpy_helper.from_array(conv_b, "conv_b"),
        numpy_helper.from_array(fc_w, "fc_w"),
        numpy_helper.from_array(fc_b, "fc_b"),
    ]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "tests/test_model.onnx")

# Also run inference to get expected output
import onnxruntime as ort
input_data = np.ones((1, 1, 8, 8), dtype=np.float32)
sess = ort.InferenceSession("tests/test_model.onnx")
result = sess.run(None, {"input": input_data})
print("Input shape:", input_data.shape)
print("Output shape:", result[0].shape)
print("Output:", result[0])
print("Sum of output (should be ~1.0):", result[0].sum())

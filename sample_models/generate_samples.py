"""Generate verified minimal sample ONNX models for testing and demonstration."""

import os
import numpy as np
import onnx
from onnx import TensorProto, helper


def generate_linear_classifier(output_path: str) -> str:
    """Generate a 2-layer MLP classifier in ONNX format."""
    # Inputs & Outputs
    x = helper.make_tensor_value_info("input_features", TensorProto.FLOAT, ["batch_size", 16])
    y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch_size", 3])

    # Weights
    w1_val = np.random.randn(16, 32).astype(np.float32) * 0.1
    b1_val = np.zeros(32, dtype=np.float32)
    w2_val = np.random.randn(32, 3).astype(np.float32) * 0.1
    b2_val = np.zeros(3, dtype=np.float32)

    w1_init = helper.make_tensor("w1", TensorProto.FLOAT, [16, 32], w1_val.flatten().tolist())
    b1_init = helper.make_tensor("b1", TensorProto.FLOAT, [32], b1_val.flatten().tolist())
    w2_init = helper.make_tensor("w2", TensorProto.FLOAT, [32, 3], w2_val.flatten().tolist())
    b2_init = helper.make_tensor("b2", TensorProto.FLOAT, [3], b2_val.flatten().tolist())

    # Nodes
    gemm1 = helper.make_node("Gemm", ["input_features", "w1", "b1"], ["hidden_pre"], name="dense_1", alpha=1.0, beta=1.0)
    relu1 = helper.make_node("Relu", ["hidden_pre"], ["hidden_act"], name="relu_1")
    gemm2 = helper.make_node("Gemm", ["hidden_act", "w2", "b2"], ["logits"], name="dense_2", alpha=1.0, beta=1.0)
    softmax = helper.make_node("Softmax", ["logits"], ["probabilities"], name="softmax_out", axis=1)

    graph = helper.make_graph(
        nodes=[gemm1, relu1, gemm2, softmax],
        name="LinearMLPClassifier",
        inputs=[x],
        outputs=[y],
        initializer=[w1_init, b1_init, w2_init, b2_init],
        doc_string="A lightweight 2-layer MLP classifier for tabular and vector feature classification.",
    )

    model = helper.make_model(
        graph,
        producer_name="Z-ModelPilot Generator",
        producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
        doc_string="Sample verified ONNX model for deployment analysis.",
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def generate_conv_net(output_path: str) -> str:
    """Generate a minimal Convolutional Vision feature extractor in ONNX format."""
    # Input: [batch_size, 1, 28, 28] (e.g. grayscale image)
    x = helper.make_tensor_value_info("image_tensor", TensorProto.FLOAT, ["batch_size", 1, 28, 28])
    y = helper.make_tensor_value_info("class_scores", TensorProto.FLOAT, ["batch_size", 10])

    # Weights: Conv 8 filters of 3x3
    w_conv = np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.1
    b_conv = np.zeros(8, dtype=np.float32)
    # Output of Conv (valid padding 28 - 2 = 26): [batch, 8, 26, 26]
    # MaxPool 2x2 with stride 2: [batch, 8, 13, 13] -> flattened = 8 * 13 * 13 = 1352
    w_fc = np.random.randn(1352, 10).astype(np.float32) * 0.1
    b_fc = np.zeros(10, dtype=np.float32)

    w_conv_init = helper.make_tensor("conv_w", TensorProto.FLOAT, [8, 1, 3, 3], w_conv.flatten().tolist())
    b_conv_init = helper.make_tensor("conv_b", TensorProto.FLOAT, [8], b_conv.flatten().tolist())
    w_fc_init = helper.make_tensor("fc_w", TensorProto.FLOAT, [1352, 10], w_fc.flatten().tolist())
    b_fc_init = helper.make_tensor("fc_b", TensorProto.FLOAT, [10], b_fc.flatten().tolist())

    conv_node = helper.make_node(
        "Conv",
        inputs=["image_tensor", "conv_w", "conv_b"],
        outputs=["conv_out"],
        name="conv2d_1",
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"], name="conv_relu")
    pool_node = helper.make_node(
        "MaxPool",
        ["relu_out"],
        ["pool_out"],
        name="max_pool_1",
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    flatten_node = helper.make_node("Flatten", ["pool_out"], ["flattened"], name="flatten_1", axis=1)
    gemm_node = helper.make_node("Gemm", ["flattened", "fc_w", "fc_b"], ["class_scores"], name="dense_head")

    graph = helper.make_graph(
        nodes=[conv_node, relu_node, pool_node, flatten_node, gemm_node],
        name="ConvFeatureExtractor",
        inputs=[x],
        outputs=[y],
        initializer=[w_conv_init, b_conv_init, w_fc_init, b_fc_init],
        doc_string="A lightweight single-stage CNN for image feature extraction and digit classification.",
    )

    model = helper.make_model(
        graph,
        producer_name="Z-ModelPilot Generator",
        producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
        doc_string="Sample vision ONNX model for architecture capability testing.",
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


if __name__ == "__main__":
    os.makedirs("sample_models", exist_ok=True)
    p1 = generate_linear_classifier(os.path.join("sample_models", "linear_classifier.onnx"))
    p2 = generate_conv_net(os.path.join("sample_models", "conv_net.onnx"))
    print(f"Generated sample models:\n - {p1}\n - {p2}")

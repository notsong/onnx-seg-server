"""Create a deterministic test-only RGB classifier ONNX. Not a trained model."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-deps"))
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_model(path, height=32, width=40, mode="rgb", dynamic=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = ("height", "width") if dynamic else (height, width)
    classes = 2 if mode == "binary" else 3
    weights = np.eye(3, dtype=np.float32)[:classes].reshape(classes, 3, 1, 1)
    initializers = [numpy_helper.from_array(weights, "weights")]
    nodes = [helper.make_node("Conv", ["image", "weights"], ["scores"])]
    result, oh, ow = "scores", h, w
    if mode == "position":
        bias = np.zeros((1, 3, height, width), np.float32)
        bias[0, 0] = np.linspace(0, 0.9, width, dtype=np.float32)[None, :]
        bias[0, 1] = 0.2
        initializers.append(numpy_helper.from_array(bias, "bias"))
        nodes.append(helper.make_node("Add", ["scores", "bias"], ["result"]))
        result = "result"
    elif mode == "geometry":
        bias = np.zeros((1, 3, height, width), np.float32)
        bias[0, 0] = np.linspace(0, 0.913, width, dtype=np.float32)[None, :]
        bias[0, 1] = np.linspace(0, 0.877, height, dtype=np.float32)[:, None]
        bias[0, 2] = 0.417
        initializers.append(numpy_helper.from_array(bias, "bias"))
        nodes.append(helper.make_node("Add", ["scores", "bias"], ["result"]))
        result = "result"
    elif mode == "lowres":
        nodes.append(helper.make_node("AveragePool", ["scores"], ["result"],
                                      kernel_shape=[2, 2], strides=[2, 2]))
        result, oh, ow = "result", height // 2, width // 2
    elif mode == "nonfinite":
        initializers += [
            numpy_helper.from_array(np.array(1, np.float32), "one"),
            numpy_helper.from_array(np.array(0, np.float32), "zero")]
        nodes += [
            helper.make_node("Sub", ["scores", "one"], ["denominator"]),
            helper.make_node("Div", ["zero", "denominator"], ["result"])]
        result = "result"
    nodes.append(helper.make_node("Identity", [result], ["output"]))
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                             [1, classes, oh, ow])]
    if mode in ("auxiliary", "ambiguous"):
        aux_classes = 1 if mode == "auxiliary" else classes
        aux_weights = np.eye(3, dtype=np.float32)[:aux_classes].reshape(aux_classes, 3, 1, 1)
        initializers.append(numpy_helper.from_array(aux_weights, "aux_weights"))
        nodes.append(helper.make_node("Conv", ["image", "aux_weights"], ["boundary"]))
        # Put the auxiliary tensor first to catch accidental output-index assumptions.
        outputs.insert(0, helper.make_tensor_value_info(
            "boundary", TensorProto.FLOAT, [1, aux_classes, h, w]))
    graph = helper.make_graph(nodes, "test_rgb_classifier",
                              [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                                            [1, 3, h, w])],
                              outputs, initializers)
    model = helper.make_model(graph, producer_name="onnx-seg-server self-test",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "demo_rgb.onnx")
    args = parser.parse_args()
    print("Created test-only model:", make_model(args.output))
    from PIL import Image
    demo = np.zeros((73, 91, 3), np.uint8)
    demo[:, :30, 0] = 255
    demo[:, 30:60, 1] = 255
    demo[:, 60:, 2] = 255
    image_path = ROOT / "tests" / "demo_input.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(demo).save(image_path)
    print("Created demo image:", image_path)

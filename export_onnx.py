"""python export_onnx.py tb  -> weights/tb.onnx + tb.int8.onnx (offline mobile inference)"""
import sys, torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from core.models import load
d = sys.argv[1]; m, _ = load(d)
torch.onnx.export(m, torch.randn(1, 3, 300, 300), f"weights/{d}.onnx", input_names=["x"], dynamic_axes={"x": {0: "b"}})
quantize_dynamic(f"weights/{d}.onnx", f"weights/{d}.int8.onnx", weight_type=QuantType.QInt8)

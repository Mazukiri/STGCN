"""
Export ST-GCN to ONNX and quantize for Android deployment.

Before running:
    uv add onnxruntime

Usage:
    uv run python src/export_model.py

Outputs to android_assets/:
    stgcn.onnx          - full precision (~7 MB)
    stgcn_quant.onnx    - INT8 dynamic quantized (~2 MB)
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.stgcn import STGCN


def export():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", "best_stgcn_model.pth")
    mapping_path = os.path.join(script_dir, "..", "data", "processed", "label_mapping.json")
    out_dir = os.path.join(script_dir, "..", "android_assets")
    os.makedirs(out_dir, exist_ok=True)

    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    num_classes = len(mapping)
    print(f"[*] Classes: {num_classes}")

    model = STGCN(in_channels=3, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print("[*] Weights loaded.")

    example = torch.randn(1, 3, 60, 75)

    # FP32 ONNX — use legacy exporter (dynamo=False) to get a self-contained single file.
    # PyTorch 2.2 dynamo exporter creates external .data files which ONNX Runtime Android
    # cannot load from the assets directory.
    onnx_path = os.path.join(out_dir, "stgcn.onnx")
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
    )
    size_fp32 = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[*] ONNX exported: {onnx_path} ({size_fp32:.1f} MB)")

    # Verify FP32
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": example.numpy()})[0]
    assert onnx_out.shape == (1, num_classes), f"Bad shape: {onnx_out.shape}"

    with torch.no_grad():
        pt_out = model(example).numpy()
    np.testing.assert_allclose(pt_out, onnx_out, atol=1e-4)
    print("[*] FP32 ONNX matches PyTorch output (atol=1e-4).")

    # INT8 dynamic quantization
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quant_path = os.path.join(out_dir, "stgcn_quant.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
    size_quant = os.path.getsize(quant_path) / 1024 / 1024
    print(f"[*] Quantized: {quant_path} ({size_quant:.1f} MB)")

    # Verify quantized top-1 matches FP32
    sess_q = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
    quant_out = sess_q.run(None, {"input": example.numpy()})[0]
    pt_top1 = int(np.argmax(pt_out))
    q_top1 = int(np.argmax(quant_out))
    match = "MATCH" if pt_top1 == q_top1 else "MISMATCH — check accuracy before shipping"
    print(f"[*] Top-1: PyTorch={pt_top1}, Quantized={q_top1} [{match}]")

    # Copy supporting assets
    import shutil

    for src, dst in [
        (os.path.join(script_dir, "..", "data", "models", "pose_landmarker_lite.task"),
         os.path.join(out_dir, "pose_landmarker_lite.task")),
        (os.path.join(script_dir, "..", "data", "models", "hand_landmarker.task"),
         os.path.join(out_dir, "hand_landmarker.task")),
        (mapping_path, os.path.join(out_dir, "label_mapping.json")),
    ]:
        shutil.copy2(src, dst)
        print(f"[*] Copied: {os.path.basename(dst)}")

    print("\n[+] Done. Copy android_assets/ → android/app/src/main/assets/")


if __name__ == "__main__":
    export()

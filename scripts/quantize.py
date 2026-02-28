# src/quantize.py
# Converts newmodel.onnx → newmodel_int8.onnx (INT8 quantization)
# Run this AFTER training. The quantized model is 2-4x faster on Pi.

import os
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(script_dir, ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

INPUT_MODEL  = os.path.join(MODELS_DIR, "newmodel.onnx")
OUTPUT_MODEL = os.path.join(MODELS_DIR, "newmodel_int8.onnx")

if not os.path.exists(INPUT_MODEL):
    raise FileNotFoundError(f"Model not found: {INPUT_MODEL}\nRun train.py first.")

# ---------------------------------------------------------------------------
# Dynamic INT8 quantization (no calibration data needed)
# ---------------------------------------------------------------------------
print(f"Quantizing {INPUT_MODEL} ...")
quantize_dynamic(
    INPUT_MODEL,
    OUTPUT_MODEL,
    weight_type=QuantType.QInt8
)
print(f"Quantized model saved → {OUTPUT_MODEL}")

# ---------------------------------------------------------------------------
# Compare model sizes
# ---------------------------------------------------------------------------
orig_size  = os.path.getsize(INPUT_MODEL)  / 1024 / 1024
quant_size = os.path.getsize(OUTPUT_MODEL) / 1024 / 1024
print(f"\nOriginal model:   {orig_size:.2f} MB")
print(f"Quantized model:  {quant_size:.2f} MB")
print(f"Size reduction:   {(1 - quant_size/orig_size)*100:.1f}%")

# ---------------------------------------------------------------------------
# Quick latency benchmark (run 20 times, report average)
# ---------------------------------------------------------------------------
print("\nRunning latency benchmark (20 inferences)...")
import time

dummy = np.random.randn(1, 2500, 12).astype(np.float32)

# Original
sess_orig = ort.InferenceSession(INPUT_MODEL)
inp_name  = sess_orig.get_inputs()[0].name
times_orig = []
for _ in range(20):
    t0 = time.perf_counter()
    sess_orig.run(None, {inp_name: dummy})
    times_orig.append((time.perf_counter() - t0) * 1000)

# Quantized
sess_quant  = ort.InferenceSession(OUTPUT_MODEL)
inp_name_q  = sess_quant.get_inputs()[0].name
times_quant = []
for _ in range(20):
    t0 = time.perf_counter()
    sess_quant.run(None, {inp_name_q: dummy})
    times_quant.append((time.perf_counter() - t0) * 1000)

print(f"\nOriginal  avg latency: {np.mean(times_orig):.1f}ms")
print(f"Quantized avg latency: {np.mean(times_quant):.1f}ms")
print(f"Speedup:               {np.mean(times_orig)/np.mean(times_quant):.2f}x")
print(f"\n✅ Use newmodel_int8.onnx for deployment on Raspberry Pi.")

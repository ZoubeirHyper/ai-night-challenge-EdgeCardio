# src/infer_onnx.py
import os
import time
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(script_dir, ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
THRESHOLD   = 0.5

# ---------------------------------------------------------------------------
# Load ONNX model
# ---------------------------------------------------------------------------
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "newmodel_int8.onnx")
if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4   # use all Pi cores
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options)
print(f"ONNX model loaded: {ONNX_MODEL_PATH}\n")

# ---------------------------------------------------------------------------
# Prediction function (single ECG or batch)
# ---------------------------------------------------------------------------
def predict_ecg(ecg_data, threshold=THRESHOLD):
    """
    ecg_data: numpy array of shape (2500, 12) or (batch, 2500, 12)
    Returns:
        probs:      sigmoid probabilities  (batch, 5)
        pred_labels: predicted class names per sample
        latency_ms: inference time in milliseconds
    """
    if ecg_data.ndim == 2:
        ecg_data = np.expand_dims(ecg_data, axis=0)

    # Per-sample normalization
    maxvals  = np.max(np.abs(ecg_data), axis=(1, 2), keepdims=True)
    maxvals  = np.where(maxvals == 0, 1.0, maxvals)
    ecg_data = ecg_data / maxvals

    inputs = {ort_session.get_inputs()[0].name: ecg_data.astype(np.float32)}

    t0      = time.perf_counter()
    logits  = ort_session.run(None, inputs)[0]
    latency = (time.perf_counter() - t0) * 1000   # ms

    probs  = 1 / (1 + np.exp(-logits))             # sigmoid
    preds  = (probs >= threshold).astype(int)

    pred_labels = []
    for row in preds:
        labels = [CLASS_NAMES[i] for i, v in enumerate(row) if v == 1]
        pred_labels.append(labels if labels else ["UNCERTAIN"])

    return probs, pred_labels, latency

# ---------------------------------------------------------------------------
# Demo: run on test samples and show predictions vs ground truth
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    CLASS_MAP = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}

    NUM_SAMPLES = 10   # change this to see more examples
    print(f"Running inference on {NUM_SAMPLES} samples...\n")
    print(f"{'#':<4} {'True Label':<10} {'Predicted':<25} {'Confidence':<35} {'Latency':>10}")
    print("-" * 90)

    latencies = []
    for i in range(NUM_SAMPLES):
        ecg       = X_test[i]
        true_lbl  = CLASS_MAP.get(int(y_test[i]), str(y_test[i])) if y_test.ndim == 1 else \
                    [CLASS_NAMES[j] for j, v in enumerate(y_test[i]) if v == 1]

        probs, pred_labels, lat = predict_ecg(ecg)
        latencies.append(lat)

        conf_str = "  ".join([f"{CLASS_NAMES[j]}:{probs[0,j]:.2f}" for j in range(5)])
        print(f"{i:<4} {str(true_lbl):<10} {str(pred_labels[0]):<25} {conf_str:<35} {lat:>8.1f}ms")

    print("-" * 90)
    print(f"\nAverage latency: {np.mean(latencies):.1f}ms")
    print(f"Max latency:     {np.max(latencies):.1f}ms")
    print(f"✅ Under 200ms:  {'YES' if np.mean(latencies) < 200 else 'NO — needs optimization'}")

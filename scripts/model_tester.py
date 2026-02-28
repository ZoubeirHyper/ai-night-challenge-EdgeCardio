# src/model_tester.py
import os
import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, classification_report

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
THRESHOLD   = 0.5   # sigmoid threshold for binary prediction

# ---------------------------------------------------------------------------
# Load ONNX model
# ---------------------------------------------------------------------------
onnx_path = os.path.join(MODELS_DIR, "newmodel_int8.onnx")
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4   # good for Pi's 4 cores
ort_session = ort.InferenceSession(onnx_path, sess_options)
print(f"ONNX model loaded: {onnx_path}\n")

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Per-sample normalization (FIX: consistent with training)
maxvals = np.max(np.abs(X_test), axis=(1, 2), keepdims=True)
maxvals = np.where(maxvals == 0, 1.0, maxvals)
X_test  = X_test / maxvals

# If Y_test is single-label integer → convert to multi-label one-hot
NUM_CLASSES = len(CLASS_NAMES)
if Y_test.ndim == 1:
    one_hot = np.zeros((len(Y_test), NUM_CLASSES), dtype=np.float32)
    one_hot[np.arange(len(Y_test)), Y_test.astype(int)] = 1.0
    Y_test = one_hot

# ---------------------------------------------------------------------------
# Run inference in batches
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
all_probs = []

for i in range(0, len(X_test), BATCH_SIZE):
    X_batch = X_test[i:i+BATCH_SIZE].astype(np.float32)
    inputs  = {ort_session.get_inputs()[0].name: X_batch}
    logits  = ort_session.run(None, inputs)[0]
    probs   = 1 / (1 + np.exp(-logits))   # sigmoid
    all_probs.append(probs)

all_probs = np.vstack(all_probs)           # (N, 5)
all_preds = (all_probs >= THRESHOLD).astype(int)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
print("=" * 50)
print("          TEST SET EVALUATION")
print("=" * 50)

# Macro-AUC (primary competition metric)
try:
    macro_auc = roc_auc_score(Y_test, all_probs, average='macro')
    print(f"\n✅ Macro-AUC (competition metric): {macro_auc:.4f}")
except ValueError as e:
    print(f"Macro-AUC error: {e}")

# Per-class AUC
print("\nPer-class AUC:")
for i, name in enumerate(CLASS_NAMES):
    try:
        auc = roc_auc_score(Y_test[:, i], all_probs[:, i])
        print(f"  {name:6s}: {auc:.4f}")
    except ValueError:
        print(f"  {name:6s}: N/A")

# Classification report
print("\nClassification Report (threshold=0.5):")
print(classification_report(Y_test, all_preds, target_names=CLASS_NAMES, zero_division=0))

print(f"Model file size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")

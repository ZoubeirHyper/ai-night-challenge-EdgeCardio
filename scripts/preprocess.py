# scripts/preprocess.py
# ---------------------------------------------------------------
# PTB-XL Dataset Preprocessing Script
# Converts raw PTB-XL files into X_train/val/test.npy and y_train/val/test.npy
#
# Usage:
#   python preprocess.py --data_dir /path/to/ptb-xl --out_dir ../data
# ---------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# The 5 diagnostic superclasses
# ---------------------------------------------------------------------------
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Preprocess PTB-XL dataset")
parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the root PTB-XL folder (contains ptbxl_database.csv)")
parser.add_argument("--out_dir", type=str, default="../data",
                    help="Output directory for .npy files (default: ../data)")
parser.add_argument("--sampling_rate", type=int, default=100,
                    help="Sampling rate: 100 or 500 Hz (default: 100)")
parser.add_argument("--val_size", type=float, default=0.1,
                    help="Validation set fraction (default: 0.1)")
parser.add_argument("--test_size", type=float, default=0.1,
                    help="Test set fraction (default: 0.1)")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
print(f"\n{'='*55}")
print(f"  PTB-XL Preprocessor")
print(f"{'='*55}")
print(f"  Data dir      : {args.data_dir}")
print(f"  Output dir    : {args.out_dir}")
print(f"  Sampling rate : {args.sampling_rate} Hz")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Step 1: Load metadata CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(args.data_dir, "ptbxl_database.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"ptbxl_database.csv not found at {csv_path}\n"
        f"Make sure --data_dir points to the root PTB-XL folder."
    )

print("Loading metadata CSV...")
df = pd.read_csv(csv_path, index_col="ecg_id")
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
print(f"  Total records: {len(df)}")

# ---------------------------------------------------------------------------
# Step 2: Load SCP statements (maps diagnosis codes → superclasses)
# ---------------------------------------------------------------------------
scp_path = os.path.join(args.data_dir, "scp_statements.csv")
if not os.path.exists(scp_path):
    raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")

scp_df = pd.read_csv(scp_path, index_col=0)
# Keep only diagnostic statements
scp_df = scp_df[scp_df.diagnostic == 1]

def get_superclass(scp_codes):
    """Map a record's scp_codes dict to a list of superclasses."""
    labels = set()
    for code in scp_codes.keys():
        if code in scp_df.index:
            superclass = scp_df.loc[code, "diagnostic_class"]
            if superclass in SUPERCLASSES:
                labels.add(superclass)
    return list(labels)

df["superclass"] = df.scp_codes.apply(get_superclass)

# Drop records with no recognized superclass
df = df[df.superclass.map(len) > 0]
print(f"  Records with valid superclass labels: {len(df)}")

# ---------------------------------------------------------------------------
# Step 3: Encode labels as multi-hot vectors
# ---------------------------------------------------------------------------
mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
Y = mlb.fit_transform(df.superclass).astype(np.float32)
print(f"\nLabel distribution (multi-label):")
for i, cls in enumerate(SUPERCLASSES):
    count = Y[:, i].sum()
    print(f"  {cls:6s}: {int(count):5d} ({count/len(Y)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Step 4: Load ECG signals
# ---------------------------------------------------------------------------
if args.sampling_rate == 100:
    signal_folder = os.path.join(args.data_dir, "records100")
    filename_col  = "filename_lr"
else:
    signal_folder = os.path.join(args.data_dir, "records500")
    filename_col  = "filename_hr"

print(f"\nLoading ECG signals at {args.sampling_rate}Hz...")
print("  This may take a few minutes...")

signals = []
valid_idx = []

for i, (ecg_id, row) in enumerate(df.iterrows()):
    if i % 1000 == 0:
        print(f"  {i}/{len(df)} records loaded...")

    try:
        filepath = os.path.join(args.data_dir, row[filename_col])
        record   = wfdb.rdsamp(filepath)
        signal   = record[0]   # shape: (samples, 12)

        # Expected: 1000 samples at 100Hz or 5000 at 500Hz for 10s
        expected = args.sampling_rate * 10
        if signal.shape[0] < expected:
            # Pad with zeros if too short
            pad = np.zeros((expected - signal.shape[0], 12))
            signal = np.vstack([signal, pad])
        else:
            signal = signal[:expected]   # trim to exact length

        signals.append(signal.astype(np.float32))
        valid_idx.append(i)

    except Exception as e:
        # skip corrupted files
        continue

print(f"  Successfully loaded: {len(signals)} records")

X = np.stack(signals)          # (N, 1000 or 5000, 12)
Y = Y[valid_idx]               # align labels with valid signals

print(f"\nData shapes:")
print(f"  X: {X.shape}   (samples, timesteps, leads)")
print(f"  Y: {Y.shape}   (samples, 5 superclasses)")

# ---------------------------------------------------------------------------
# Step 5: Train / Val / Test split
# ---------------------------------------------------------------------------
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X, Y,
    test_size=args.test_size,
    random_state=42,
    stratify=None   
)

val_fraction = args.val_size / (1 - args.test_size)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval,
    test_size=val_fraction,
    random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train : {len(X_train)}")
print(f"  Val   : {len(X_val)}")
print(f"  Test  : {len(X_test)}")

# ---------------------------------------------------------------------------
# Step 6: Save .npy files
# ---------------------------------------------------------------------------
print(f"\nSaving .npy files to {args.out_dir} ...")
np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
np.save(os.path.join(args.out_dir, "y_train.npy"), Y_train)
np.save(os.path.join(args.out_dir, "X_val.npy"),   X_val)
np.save(os.path.join(args.out_dir, "y_val.npy"),   Y_val)
np.save(os.path.join(args.out_dir, "X_test.npy"),  X_test)
np.save(os.path.join(args.out_dir, "y_test.npy"),  Y_test)

print("\n✅ Preprocessing complete!")
print(f"   X_train.npy  {X_train.shape}")
print(f"   y_train.npy  {Y_train.shape}")
print(f"   X_val.npy    {X_val.shape}")
print(f"   y_val.npy    {Y_val.shape}")
print(f"   X_test.npy   {X_test.shape}")
print(f"   y_test.npy   {Y_test.shape}")
print(f"\nNext step: python train.py")

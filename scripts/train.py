# src/train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model import myCNN
import torch.onnx
if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Directories
    # ---------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR   = os.path.abspath(os.path.join(script_dir, ".."))
    DATA_DIR   = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    NUM_CLASSES  = 5
    CLASS_NAMES  = ["NORM", "MI", "STTC", "CD", "HYP"]
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 60
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 8          # early stopping patience
    THRESHOLD    = 0.5        # sigmoid threshold for binary prediction

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}    Y_val:   {Y_val.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    # ---------------------------------------------------------------------------
    # Detect label format: single-label (1D int) vs multi-label (2D float)
    # If Y is 1D integer → convert to one-hot multi-label float
    # If Y is already 2D → use as-is
    # ---------------------------------------------------------------------------
    def to_multilabel(Y, num_classes):
        if Y.ndim == 1:
            # Single-label integer → one-hot
            one_hot = np.zeros((len(Y), num_classes), dtype=np.float32)
            one_hot[np.arange(len(Y)), Y.astype(int)] = 1.0
            return one_hot
        return Y.astype(np.float32)

    Y_train = to_multilabel(Y_train, NUM_CLASSES)
    Y_val   = to_multilabel(Y_val,   NUM_CLASSES)
    Y_test  = to_multilabel(Y_test,  NUM_CLASSES)

    # ---------------------------------------------------------------------------
    # Per-sample normalization  (FIX: was global norm on test set)
    # ---------------------------------------------------------------------------
    def per_sample_norm(X):
        maxvals = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        maxvals = np.where(maxvals == 0, 1.0, maxvals)   # avoid div-by-zero
        return X / maxvals

    X_train = per_sample_norm(X_train)
    X_val   = per_sample_norm(X_val)
    X_test  = per_sample_norm(X_test)

    # ---------------------------------------------------------------------------
    # Tensors & DataLoaders
    # ---------------------------------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(X_test_t,  Y_test_t),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ---------------------------------------------------------------------------
    # Model, loss, optimizer
    # ---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = myCNN(input_channels=12, num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # BCEWithLogitsLoss = sigmoid + binary cross-entropy — correct for multi-label
    pos_weight = torch.tensor([1.0, 1.0, 1.0, 10.0, 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.4, patience=3)

    # ---------------------------------------------------------------------------
    # Helper: compute Macro-AUC over a DataLoader
    # ---------------------------------------------------------------------------
    def compute_macro_auc(loader, model, device):
        model.eval()
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits  = model(X_batch)
                probs   = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.numpy())
        all_probs  = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        try:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
        except ValueError:
            auc = 0.0   # can happen if a class has no positive samples in batch
        return auc

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    best_val_auc     = 0.0
    best_model_path  = os.path.join(MODELS_DIR, "newmodel.pt")
    patience_counter = 0

    print("\n--- Training ---")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation AUC
        val_auc = compute_macro_auc(val_loader, model, device)
        scheduler.step(val_auc)

        # Early stopping on val AUC
        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Macro-AUC: {val_auc:.4f}  {'⭐ best' if patience_counter == 0 else ''}")

    # ---------------------------------------------------------------------------
    # Evaluate best model on test set
    # ---------------------------------------------------------------------------
    print(f"\nBest Val Macro-AUC: {best_val_auc:.4f}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_auc = compute_macro_auc(test_loader, model, device)
    print(f"Test  Macro-AUC:    {test_auc:.4f}")

    # Per-class AUC
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y_batch.numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    print("\nPer-class AUC:")
    for i, name in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            print(f"  {name}: {auc:.4f}")
        except ValueError:
            print(f"  {name}: N/A (no positive samples)")

    # ---------------------------------------------------------------------------
    # Export to ONNX
    # ---------------------------------------------------------------------------
    model.eval()
    onnx_path   = os.path.join(MODELS_DIR, "newmodel.onnx")
    dummy_input = torch.randn(1, 2500, 12).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"\nONNX model exported → {onnx_path}")

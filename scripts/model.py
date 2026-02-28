# src/model.py
import torch
import torch.nn as nn


class myCNN(nn.Module):
    """
    Lightweight 1D CNN for 12-lead ECG multi-label classification.
    Input:  (batch, 2500, 12)
    Output: (batch, 5)  — raw logits, apply sigmoid for probabilities
    Classes: NORM, MI, STTC, CD, HYP
    """

    def __init__(self, input_channels=12, num_classes=5, dropout_rate=0.3):
        super(myCNN, self).__init__()

        # --- Convolutional blocks ---
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)

        # Extra conv block for richer features
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # Global average pooling → removes dependency on sequence length
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --- Classifier head ---
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        # NOTE: No sigmoid here — BCEWithLogitsLoss handles it during training.
        # For inference, apply torch.sigmoid(output) or use predict_proba().

    def forward(self, x):
        # x: (batch, seq_len, channels) → (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))   # no pooling here
        x = self.dropout(x)

        x = self.global_pool(x)          # (batch, 128, 1)
        x = x.view(x.size(0), -1)        # (batch, 128)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                  # raw logits (batch, num_classes)
        return x

    def predict_proba(self, x):
        """Return sigmoid probabilities — use this for AUC computation."""
        return torch.sigmoid(self.forward(x))

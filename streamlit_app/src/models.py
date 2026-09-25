"""
PyTorch Neural Network Architectures for Intrusion Detection (SECIoHT-FL).
Reconstructed from Mosaiyebzadeh et al. (Electronics 2025).
"""
import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DNN(nn.Module):
        """
        Deep Neural Network baseline.
        Architecture: Linear(input_dim -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 2)
        Note: The paper's notation '64(8x8)' is implemented as a 64-unit linear layer.
        """
        def __init__(self, input_dim: int = 36, output_dim: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    class CNN(nn.Module):
        """
        1D Convolutional Neural Network baseline for tabular flow signals.
        Architecture: 4 Conv1d layers (128 -> 64 -> 32 -> 16) with AdaptiveAvgPool1d and Linear head.
        """
        def __init__(self, input_dim: int = 36, output_dim: int = 2):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, output_dim)

        def forward(self, x):
            # x: (batch, input_dim) -> (batch, 1, input_dim)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

else:
    class DNN:
        def __init__(self, input_dim: int = 36, output_dim: int = 2):
            self.input_dim = input_dim
            self.output_dim = output_dim

    class CNN:
        def __init__(self, input_dim: int = 36, output_dim: int = 2):
            self.input_dim = input_dim
            self.output_dim = output_dim


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

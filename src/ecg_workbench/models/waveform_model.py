"""Compact reference-location waveform classifier."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from ecg_workbench.data.labels import CLASS_ORDER


@dataclass(frozen=True)
class WindowNormalizer:
    mean: float
    std: float

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (np.asarray(windows, dtype=np.float32) - self.mean) / self.std


def fit_window_normalizer(windows: np.ndarray) -> WindowNormalizer:
    if windows.ndim != 2 or not np.isfinite(windows).all():
        raise ValueError("training windows must be a finite matrix")
    return WindowNormalizer(float(np.mean(windows)), max(float(np.std(windows)), 1e-6))


class BeatCNN(nn.Module):
    def __init__(self, window_samples: int = 234):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predict_waveform(model: BeatCNN, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[start : start + batch_size], dtype=torch.float32)
            outputs.append(model(batch[:, None, :]).argmax(dim=1).numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=int)


def train_waveform_model(
    train_windows: np.ndarray, train_labels: np.ndarray,
    validation_windows: np.ndarray, validation_labels: np.ndarray,
    seed: int, epochs: int = 8,
) -> BeatCNN:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = BeatCNN(train_windows.shape[1])
    counts = np.bincount(train_labels, minlength=3)
    weights = counts.sum() / (3 * np.maximum(counts, 1))
    weights = np.minimum(weights, 15)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.as_tensor(train_windows, dtype=torch.float32)
    y = torch.as_tensor(train_labels, dtype=torch.long)
    best_score = -1.0
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for batch_indices in torch.randperm(len(x)).split(256):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch_indices, None, :]), y[batch_indices])
            loss.backward()
            optimizer.step()
        predicted = predict_waveform(model, validation_windows)
        score = f1_score(validation_labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        if score > best_score + 1e-5:
            best_score = score
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.validation_macro_f1 = float(best_score)
    model.class_order = CLASS_ORDER
    return model

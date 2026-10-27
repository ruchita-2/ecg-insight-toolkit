import numpy as np
import pytest
import torch

from ecg_workbench.models.waveform_model import BeatCNN, fit_window_normalizer


def test_cnn_returns_three_logits():
    model = BeatCNN(window_samples=234)
    assert model(torch.zeros(4, 1, 234)).shape == (4, 3)


def test_normalization_uses_training_windows_only():
    stats = fit_window_normalizer(np.ones((5, 234)))
    assert stats.mean == pytest.approx(1.0)
    assert stats.std > 0

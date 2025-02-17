import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.utils.training import (
    pretrain_autoencoder,
    compute_reconstruction_error,
    train_anomaly_detector,
    evaluate_model
)


class DummyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 10)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


@pytest.fixture
def dummy_data():
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def dummy_model():
    return DummyAutoencoder()


def test_pretrain_autoencoder(dummy_data, dummy_model):
    train_loader, val_loader = dummy_data, dummy_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = pretrain_autoencoder(dummy_model, train_loader, val_loader, epochs=2, learning_rate=0.001,
                                         device=device)

    assert isinstance(trained_model, nn.Module)
    assert next(trained_model.parameters()).is_cuda == torch.cuda.is_available()


def test_compute_reconstruction_error(dummy_data, dummy_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    errors = compute_reconstruction_error(dummy_model, dummy_data, device)

    assert isinstance(errors, np.ndarray)
    assert len(errors) == len(dummy_data.dataset)


def test_train_anomaly_detector(dummy_data, dummy_model):
    train_loader, val_loader = dummy_data, dummy_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model, threshold = train_anomaly_detector(dummy_model, train_loader, val_loader, epochs=2,
                                                      learning_rate=0.001, device=device)

    assert isinstance(trained_model, nn.Module)
    assert isinstance(threshold, float)


def test_evaluate_model(dummy_data, dummy_model):
    test_loader = dummy_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 0.5

    metrics = evaluate_model(dummy_model, test_loader, threshold, device)

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())


if __name__ == "__main__":
    pytest.main()
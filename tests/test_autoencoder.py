import pytest
import torch
from src.models.autoencoder import VideoAutoencoder

@pytest.fixture
def sample_input():
    return torch.randn(1, 3, 16, 64, 64)

@pytest.fixture
def model():
    return VideoAutoencoder()

def test_autoencoder_shapes(sample_input, model):
    encoded, decoded = model(sample_input)
    assert encoded.shape == (1, 256), "Encoded shape is incorrect"
    assert decoded.shape == sample_input.shape, "Decoded shape doesn't match input"

def test_encode_decode(sample_input, model):
    encoded = model.encode(sample_input)
    decoded = model.decode(encoded)
    assert decoded.shape == sample_input.shape, "Encode-decode shape mismatch"

def test_autoencoder_output_range(sample_input, model):
    _, decoded = model(sample_input)
    assert torch.all(decoded >= 0) and torch.all(decoded <= 1), "Output values out of range [0, 1]"

def test_latent_dim(model):
    assert model.latent_dim == 256, "Unexpected latent dimension"

def test_input_channels(model):
    assert model.input_channels == 3, "Unexpected number of input channels"
"""
Video Anomaly Detection Models

This package contains neural network models used for video anomaly detection.

Current models:
- VideoAutoencoder: A convolutional autoencoder for encoding and decoding video data.

Future models (planned):
- BinaryClassifier: A classifier for determining if a video contains anomalies.

Version: 0.1.0
Author: Your Name
License: MIT
"""

import torch
from typing import Dict, Any

# Import models
from .autoencoder import VideoAutoencoder

# If you add more models in the future, import them here
# from .classifier import BinaryClassifier

# Explicitly declare public exports
__all__ = ['VideoAutoencoder', 'get_model', 'list_available_models']

# Version of the models package
__version__ = '0.1.0'

# Dictionary to store model classes
AVAILABLE_MODELS: Dict[str, Any] = {
    'VideoAutoencoder': VideoAutoencoder,
    # Add more models here as they are implemented
    # 'BinaryClassifier': BinaryClassifier,
}


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    """
    Factory function to get a model instance by name.

    Args:
        model_name (str): Name of the model to instantiate.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        torch.nn.Module: An instance of the requested model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models are: {', '.join(AVAILABLE_MODELS.keys())}")

    return AVAILABLE_MODELS[model_name](**kwargs)


def list_available_models() -> Dict[str, str]:
    """
    List all available models with their descriptions.

    Returns:
        Dict[str, str]: A dictionary where keys are model names and values are their docstrings.
    """
    return {name: model.__doc__ for name, model in AVAILABLE_MODELS.items()}

# You can add more utility functions here as needed, for example:
# def save_model(model: torch.nn.Module, path: str) -> None:
#     """Save a model to disk."""
#     torch.save(model.state_dict(), path)

# def load_model(model_name: str, path: str) -> torch.nn.Module:
#     """Load a model from disk."""
#     model = get_model(model_name)
#     model.load_state_dict(torch.load(path))
#     return model
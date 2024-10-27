"""
Utility training functions

This package contains utility functions for model training and evaluation
such as pretraining autoencoder, computing reconstruction errors, and
evaluating model performance.

Version: 0.1.0
Author: RocketSteve
License: MIT
"""
from .training import (
    pretrain_autoencoder,
    compute_reconstruction_error,
    evaluate_model
)

__all__ = [
    # Training utilities
    'pretrain_autoencoder',
    'compute_reconstruction_error',
    'evaluate_model'
]

__version__ = '0.1.0'
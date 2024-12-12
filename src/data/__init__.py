"""
Functions operation on dataset

This package contains utility functions for data division, augmentation and
clip creation.

Version: 0.1.0
Author: RocketSteve
License: MIT
"""

from .clipper import VideoClipDataset

__all__ = [
    'VideoClipDataset'
]

__version__ = '0.1.0'
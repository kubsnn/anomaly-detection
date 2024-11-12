import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import logging


class VideoAugmentation:
    """
    A class for applying various augmentations to video data.

    Includes transforms like grayscale conversion.
    """

    def __init__(self, p=0.5):
        """Initialize the VideoAugmentation class."""
        self.p = p
        self.logger = logging.getLogger(__name__)

    def convert_to_grayscale(self, video):
        """
        Convert RGB video to grayscale while maintaining 3 channels.
        Processes frame by frame to avoid OpenCV errors.

        Args:
            video (np.ndarray): Input video array of shape [T, H, W, C]

        Returns:
            np.ndarray: Grayscale video with 3 channels
        """
        gray_frames = []

        for frame in video:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # 3 channels
            gray_3ch = np.stack([gray] * 3, axis=-1)
            gray_frames.append(gray_3ch)

        return np.array(gray_frames)

    def __call__(self, video):
        """Apply random augmentations to the video."""

        if not isinstance(video, np.ndarray):
            self.logger.error(f"Expected numpy array, got {type(video)}")
            raise TypeError(f"Expected numpy array, got {type(video)}")

        self.logger.debug(f"Input video shape: {video.shape}")

        if isinstance(video, torch.Tensor):
            video = video.numpy()

        if len(video.shape) != 4:  # [T, H, W, C]
            self.logger.error(f"Expected 4D array [T, H, W, C], got shape {video.shape}")
            raise ValueError(f"Expected 4D array [T, H, W, C], got shape {video.shape}")

        if video.shape[-1] != 3:  # RGB channels
            self.logger.error(f"Expected 3 channels, got {video.shape[-1]}")
            raise ValueError(f"Expected 3 channels, got {video.shape[-1]}")

        try:
            self.logger.debug(f"Applying convert_to_grayscale")
            video = self.convert_to_grayscale(video)
        except Exception as e:
            self.logger.error(f"Error in convert_to_grayscale: {str(e)}")
            raise

        self.logger.debug(f"Output video shape: {video.shape}")
        return video
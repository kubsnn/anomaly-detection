import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import logging


class VideoAugmentation:
    """
    A class for applying various augmentations to video data.

    Includes transforms like grayscale conversion, brightness/contrast adjustment,
    geometric transformations, and noise addition.
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

    def random_brightness(self, video, delta=0.2):
        """
        Randomly adjust video brightness frame by frame.

        Args:
            video (np.ndarray): Input video array [T, H, W, C]
            delta (float): Maximum brightness change
        """
        adjusted_frames = []

        for frame in video:
            hsv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2HSV)
            random_delta = np.random.uniform(-delta, delta)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + random_delta), 0, 255)
            adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            adjusted_frames.append(adjusted)

        return np.array(adjusted_frames)

    def random_contrast(self, video, lower=0.8, upper=1.2):
        """
        Randomly adjust video contrast.

        Args:
            video (np.ndarray): Input video array
            lower (float): Lower contrast limit. Default is 0.8
            upper (float): Upper contrast limit. Default is 1.2

        Returns:
            np.ndarray: Video with adjusted contrast
        """
        alpha = np.random.uniform(lower, upper)
        return np.clip(video * alpha, 0, 255).astype(np.uint8)

    def random_flip(self, video):
        """
        Randomly flip video horizontally.

        Args:
            video (np.ndarray): Input video array

        Returns:
            np.ndarray: Horizontally flipped video
        """
        return np.flip(video, axis=2)

    def random_rotation(self, video, max_degrees=10):
        """
        Randomly rotate video frames.

        Args:
            video (np.ndarray): Input video array
            max_degrees (int): Maximum rotation angle. Default is 10

        Returns:
            np.ndarray: Rotated video
        """
        angle = np.random.uniform(-max_degrees, max_degrees)
        height, width = video.shape[1:3]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_frames = []
        for frame in video:
            rotated = cv2.warpAffine(frame, rotation_matrix, (width, height))
            rotated_frames.append(rotated)

        return np.array(rotated_frames)

    def random_crop(self, video, crop_size=(320, 320)):
        """
        Randomly crop video frames spatially.

        Args:
            video (np.ndarray): Input video array
            crop_size (tuple): Height and width of crop. Default is (320, 320)

        Returns:
            np.ndarray: Cropped video
        """
        height, width = video.shape[1:3]
        start_h = np.random.randint(0, height - crop_size[0])
        start_w = np.random.randint(0, width - crop_size[1])

        return video[:,
               start_h:start_h + crop_size[0],
               start_w:start_w + crop_size[1]]

    def temporal_crop(self, video, crop_size=16):
        """
        Randomly crop video in temporal dimension.

        Args:
            video (np.ndarray): Input video array
            crop_size (int): Number of frames to keep. Default is 16

        Returns:
            np.ndarray: Temporally cropped video
        """
        if video.shape[0] > crop_size:
            start = np.random.randint(0, video.shape[0] - crop_size)
            return video[start:start + crop_size]
        return video

    def add_noise(self, video, noise_factor=0.05):
        """
        Add random noise to video.

        Args:
            video (np.ndarray): Input video array
            noise_factor (float): Intensity of noise. Default is 0.05

        Returns:
            np.ndarray: Video with added noise
        """
        noise = np.random.normal(0, 1, video.shape) * noise_factor * 255
        noisy_video = video + noise
        return np.clip(noisy_video, 0, 255).astype(np.uint8)

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

        augmentations = [
            (self.convert_to_grayscale, 0.3),
            (self.random_brightness, self.p),
            (self.random_contrast, self.p),
            (self.random_flip, self.p),
            (self.random_rotation, self.p),
            (self.add_noise, 0.2),
        ]

        if video.shape[0] > 16:
            video = self.temporal_crop(video)

        for aug_func, prob in augmentations:
            try:
                if np.random.random() < prob:
                    self.logger.debug(f"Applying {aug_func.__name__}")
                    video = aug_func(video)
            except Exception as e:
                self.logger.error(f"Error in {aug_func.__name__}: {str(e)}")
                raise

        self.logger.debug(f"Output video shape: {video.shape}")
        return video
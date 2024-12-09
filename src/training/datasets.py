import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import logging
from pathlib import Path


class ProcessedVideoDataset(Dataset):
    def __init__(self, video_paths):
        """
        Initialize the video dataset for reconstruction training.

        Args:
            video_paths (list): List of paths to processed video files
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.logger = logging.getLogger(__name__)

        # Validate videos and get metadata
        self.videos = self._scan_videos()
        self.logger.info(f"Created dataset with {len(self.videos)} videos")

    def _scan_videos(self):
        """Scan and validate videos, collecting metadata."""
        videos = []

        for video_path in self.video_paths:
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.logger.error(f"Could not open video file: {video_path}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames == 0:
                    self.logger.warning(f"Video has 0 frames: {video_path}")
                    continue

                videos.append({
                    'path': video_path,
                    'total_frames': total_frames
                })

            except Exception as e:
                self.logger.error(f"Error scanning video {video_path}: {str(e)}")
                continue

        if not videos:
            raise ValueError("No valid videos found in the dataset")

        return videos

    def _load_video(self, video_path):
        """
        Load a video file into memory.

        Args:
            video_path (Path): Path to the video file

        Returns:
            torch.Tensor: Video tensor of shape [C, T, H, W]
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from {video_path}")

        # Stack frames and convert to float tensor
        frames = np.stack(frames)
        frames = torch.FloatTensor(frames)
        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        frames = frames / 255.0
        frames = (frames - 0.5) / 0.5

        return frames

    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Get a video tensor.

        Args:
            idx (int): Video index

        Returns:
            torch.Tensor: Video tensor of shape [C, T, H, W]
        """
        try:
            video_info = self.videos[idx]
            return self._load_video(video_info['path'])

        except Exception as e:
            self.logger.error(f"Error loading video at index {idx}: {str(e)}")
            raise


def create_dataloaders(train_paths, val_paths, test_paths, config):
    """
    Create data loaders for training, validation and testing.

    Args:
        train_paths (list): List of paths to training videos
        val_paths (list): List of paths to validation videos
        test_paths (list): List of paths to test videos
        config (dict): Configuration dictionary containing:
            - batch_size: Batch size for training

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = ProcessedVideoDataset(train_paths)
    val_dataset = ProcessedVideoDataset(val_paths)
    test_dataset = ProcessedVideoDataset(test_paths)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
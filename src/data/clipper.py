import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import logging


class ProcessedVideoDataset(Dataset):
    def __init__(self, video_paths, target_size=(64, 64)):
        """
        Dataset for preprocessed video clips.

        Args:
            video_paths (list): List of paths to preprocessed video files
            target_size (tuple): Target frame size (height, width)
        """
        self.video_paths = video_paths
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        self.clips = self._load_clips()
        self.logger.info(f"Created dataset with {len(self.clips)} clips")

    def _load_clips(self):
        """Load all preprocessed clips and their labels."""
        clips = []

        for video_path in self.video_paths:
            try:
                # Extract label from filename (N or F prefix)
                filename = os.path.basename(video_path)
                first_char = filename[0].upper()

                label = 1 if first_char == 'F' else 0

                # Verify the video can be opened
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.logger.error(f"Could not open video file: {video_path}")
                    continue
                cap.release()

                clips.append({
                    'video_path': video_path,
                    'label': label
                })

            except Exception as e:
                self.logger.error(f"Error processing {video_path}: {str(e)}")
                continue

        if not clips:
            raise ValueError("No valid clips found in the dataset")

        return clips

    def __len__(self):
        """Return the total number of clips."""
        return len(self.clips)

    def __getitem__(self, idx):
        """
        Get a video clip and its label.

        Args:
            idx (int): Clip index

        Returns:
            tuple: (frames, label)
                frames: torch.Tensor of shape [C, T, H, W]
                label: int (0 or 1)
        """
        try:
            clip_info = self.clips[idx]
            video_path = clip_info['video_path']
            label = clip_info['label']

            # Load all frames from the preprocessed clip
            frames = []
            cap = cv2.VideoCapture(str(video_path))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if not frames:
                raise ValueError(f"No frames found in {video_path}")

            # Stack frames and convert to tensor
            frames = np.stack(frames)
            frames = torch.FloatTensor(frames)
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]

            # Normalize
            frames = frames / 255.0
            frames = (frames - 0.5) / 0.5

            return frames, label

        except Exception as e:
            self.logger.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            raise
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import logging


class VideoClipDataset(Dataset):
    def __init__(self, video_paths, clip_length=16, clip_overlap=0.5, min_clips=1,
                 augment=True, target_size=(64, 64)):
        """
        Initialize the video clip dataset.

        Args:
            video_paths (list): List of paths to video files
            clip_length (int): Number of frames per clip
            clip_overlap (float): Overlap between consecutive clips (0-1)
            min_clips (int): Minimum number of clips per video
            augment (bool): Whether to apply augmentation
            target_size (tuple): Target frame size (height, width)
        """
        self.video_paths = video_paths
        self.clip_length = clip_length
        self.clip_overlap = clip_overlap
        self.min_clips = min_clips
        self.augment = augment
        self.target_size = target_size

        self.logger = logging.getLogger(__name__)

        # Initialize transform if augmentation is enabled
        if self.augment:
            from .augmentation import VideoAugmentation
            self.transform = VideoAugmentation()
        else:
            self.transform = None

        # Pre-compute clips for each video
        self.clips = self._compute_clips()
        self.logger.info(f"Created dataset with {len(self.clips)} clips from {len(video_paths)} videos")

    def _compute_clips(self):
        """
        Pre-compute clip indices for all videos.

        Returns:
            list: List of dictionaries containing clip information
        """
        all_clips = []

        for video_idx, video_path in enumerate(self.video_paths):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.logger.error(f"Could not open video file: {video_path}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames == 0:
                    self.logger.error(f"Video has 0 frames: {video_path}")
                    continue

                stride = int(self.clip_length * (1 - self.clip_overlap))

                n_clips = max(
                    self.min_clips,
                    (total_frames - self.clip_length) // stride + 1
                )

                if total_frames < self.clip_length:
                    clip_starts = [0]
                else:
                    if n_clips == 1:
                        clip_starts = [(total_frames - self.clip_length) // 2]
                    else:
                        clip_starts = np.linspace(
                            0,
                            total_frames - self.clip_length,
                            n_clips,
                            dtype=int
                        )

                for start in clip_starts:
                    all_clips.append({
                        'video_idx': video_idx,
                        'start_frame': int(start),
                        'end_frame': int(start + self.clip_length),
                        'total_frames': total_frames,
                        'video_path': video_path
                    })

            except Exception as e:
                self.logger.error(f"Error processing video {video_path}: {str(e)}")
                continue

        if not all_clips:
            raise ValueError("No valid clips found in the dataset")

        return all_clips

    def _load_clip(self, video_path, start_frame, end_frame, total_frames):
        """
        Load and resize a clip from a video file.

        Args:
            video_path (str): Path to video file
            start_frame (int): Starting frame index
            end_frame (int): Ending frame index
            total_frames (int): Total number of frames in video

        Returns:
            np.ndarray: Video clip array of shape [T, H, W, C]
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.target_size[0], self.target_size[1], 3),
                                           dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]),
                                   interpolation=cv2.INTER_AREA)
                frames.append(frame)

        cap.release()
        return np.ascontiguousarray(np.array(frames))

    def __len__(self):
        """Return the total number of clips."""
        return len(self.clips)

    def __getitem__(self, idx):
        """
        Get a video clip.

        Args:
            idx (int): Clip index

        Returns:
            torch.Tensor: Video clip tensor of shape [C, T, H, W]
        """
        try:
            clip_info = self.clips[idx]
            video_path = clip_info['video_path']

            frames = self._load_clip(
                video_path,
                clip_info['start_frame'],
                clip_info['end_frame'],
                clip_info['total_frames']
            )

            self.logger.debug(f"Loaded clip shape: {frames.shape}")

            if self.augment and self.transform is not None:
                frames = self.transform(frames)
                frames = np.ascontiguousarray(frames)

            # tensor + normalize
            frames = torch.FloatTensor(frames)
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
            frames = frames / 255.0
            frames = (frames - 0.5) / 0.5

            expected_shape = (3, self.clip_length, self.target_size[0], self.target_size[1])
            assert frames.shape == expected_shape, \
                f"Wrong shape: got {frames.shape}, expected {expected_shape}"

            return frames

        except Exception as e:
            self.logger.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            raise
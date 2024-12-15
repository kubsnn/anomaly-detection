import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import setup_logging
from tqdm import tqdm

logger = setup_logging(__name__)

class VideoClipDataset(Dataset):
    def __init__(self, video_paths, clip_length=16, clip_overlap=0.5, min_clips=1,
                 augment=True, target_size=(128, 128)):
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

        # Pre-compute clips for each video
        self.clips = self._compute_clips()
        logger.info(f"Created dataset with {len(self.clips)} clips from {len(video_paths)} videos")

    def _compute_clips(self):
        """
        Pre-compute clip indices for all videos using multithreading.

        Returns:
            list: List of dictionaries containing clip information
        """
        def process_video(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video file: {video_path}")
                    return []

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames == 0:
                    logger.error(f"Video has 0 frames: {video_path}")
                    return []

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

                filename = os.path.basename(video_path)
                first_char = filename[0].upper()

                if first_char == 'N':
                    label = 0
                elif first_char == 'F':
                    label = 1
                else:
                    logger.warning(f"Unknown label for file {filename}, defaulting to 0")
                    label = 0

                clips = []
                for start in clip_starts:
                    clips.append({
                        'video_idx': video_path,
                        'start_frame': int(start),
                        'end_frame': int(start + self.clip_length),
                        'total_frames': total_frames,
                        'video_path': video_path,
                        'label': label
                    })

                return clips

            except Exception as e:
                logger.error(f"Error processing video {video_path}: {str(e)}")
                return []

        logger.info(f"Pre-computing clips for {len(self.video_paths)} videos...")
        all_clips = []
        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust `max_workers` based on your CPU
            future_to_video = {executor.submit(process_video, video): video for video in self.video_paths}

            for future in tqdm(as_completed(future_to_video), total=len(self.video_paths), desc="Processing videos"):
                all_clips.extend(future.result())

        if not all_clips:
            raise ValueError("No valid clips found in the dataset")

        logger.info(f"Finished pre-computing {len(all_clips)} clips from {len(self.video_paths)} videos.")
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
            np.ndarray: Video clip array of shape [T, H, W, 1]
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
                    frames.append(np.zeros((self.target_size[0], self.target_size[1], 1),
                                           dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]),
                                   interpolation=cv2.INTER_AREA)
                frame = frame / 255.0  # Normalize to [0, 1]
                frame = (frame - 0.5) * 2.0  # Shift to [-1, 1]
                frame = frame[..., np.newaxis]  # Add channel dimension for grayscale
                frames.append(frame)

        cap.release()
        return np.ascontiguousarray(np.array(frames))

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
            label = clip_info['label']  # Retrieve the label

            frames = self._load_clip(
                video_path,
                clip_info['start_frame'],
                clip_info['end_frame'],
                clip_info['total_frames']
            )

            logger.debug(f"Loaded clip shape: {frames.shape}")

            # Convert to tensor and normalize
            frames = torch.FloatTensor(frames)
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
            frames = torch.clip(frames, -1, 1)  # Ensure the values remain between -1 and 1

            expected_shape = (1, self.clip_length, self.target_size[0], self.target_size[1])
            assert frames.shape == expected_shape, \
                f"Wrong shape: got {frames.shape}, expected {expected_shape}"

            return frames, label  # Return both frames and label

        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            raise

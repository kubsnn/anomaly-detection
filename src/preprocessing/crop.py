import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from tqdm import tqdm
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def video_capture(path):
    """Safely handle video opening with automatic closing"""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {path}")
        yield cap
    finally:
        cap.release()


@contextmanager
def video_writer(path, fps, size, is_color=True):
    """Safely handle video writing with automatic closing"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size, isColor=is_color)
    try:
        if not writer.isOpened():
            raise IOError(f"Failed to create video writer: {path}")
        yield writer
    finally:
        writer.release()


class VideoCropper:
    def __init__(self, target_size: Tuple[int, int] = (320, 180)):
        """
        Initialize the VideoCropper.

        Args:
            target_size (tuple): Target resolution (width, height) for the output video
        """
        self.target_size = target_size

    def detect_content_area(self, frame: np.ndarray, threshold: int = 30) -> Optional[Tuple[int, int, int, int]]:
        """Detect the main content area in a frame."""
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        target_ratio = 16 / 9

        def is_content_row(values, threshold):
            std = np.std(values)
            return std > threshold / 2

        def is_content_col(values, threshold):
            std = np.std(values)
            mean = np.mean(values)
            return std > threshold / 8 or (mean > threshold and mean < 255 - threshold)

        left = 0
        right = w - 1
        top = 0
        bottom = h - 1

        # Find left boundary
        for x in range(w // 2):
            if is_content_col(gray[:, x], threshold):
                left = x
                break

        # Find right boundary
        for x in range(w - 1, w // 2, -1):
            if is_content_col(gray[:, x], threshold):
                right = x
                break

        # Find top boundary
        for y in range(h // 2):
            if is_content_row(gray[y, :], threshold):
                top = y
                break

        # Find bottom boundary
        for y in range(h - 1, h // 2, -1):
            if is_content_row(gray[y, :], threshold):
                bottom = y
                break

        content_width = right - left
        content_height = bottom - top
        current_ratio = content_width / content_height

        # Force 16:9 ratio by cropping excess content
        if current_ratio < target_ratio:
            # Video is too tall 4:3 - crop height
            required_height = int(content_width / target_ratio)
            excess_height = content_height - required_height
            top += excess_height // 2
            bottom -= excess_height // 2
        else:
            # Video is too wide - crop width
            required_width = int(content_height * target_ratio)
            excess_width = content_width - required_width
            left += excess_width // 2
            right -= excess_width // 2

        # Add a small margin to ensure no white edges
        margin = 2
        left = left + margin
        right = right - margin
        top = top + margin
        bottom = bottom - margin

        return (left, right, top, bottom)

    def crop_video(self, video_path: Path, output_path: Path) -> Path:
        """
        Crop the video to focus on the main content area and resize to target resolution.

        Args:
            video_path (Path): Input video path
            output_path (Path): Output video path

        Returns:
            Path: Path to the processed video
        """
        if output_path.exists():
            logger.info(f"Skipping existing cropped video: {output_path}")
            return output_path

        # First detect content area
        with video_capture(video_path) as cap:
            sample_frames = 5
            frame_positions = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, sample_frames, dtype=int)
            lefts, rights, tops, bottoms = [], [], [], []

            for pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    coords = self.detect_content_area(frame)
                    if coords:
                        left, right, top, bottom = coords
                        lefts.append(left)
                        rights.append(right)
                        tops.append(top)
                        bottoms.append(bottom)

            if not lefts:
                raise ValueError(f"Could not detect content area in {video_path}")

            # Use median coordinates
            left = int(np.median(lefts))
            right = int(np.median(rights))
            top = int(np.median(tops))
            bottom = int(np.median(bottoms))

        # Create temporary file for cropped video
        temp_dir = output_path.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_cropped = temp_dir / f"temp_cropped_{video_path.name}"

        try:
            # First pass: Crop the video
            with video_capture(video_path) as cap, \
                    video_writer(temp_cropped, int(cap.get(cv2.CAP_PROP_FPS)),
                                 (right - left, bottom - top), is_color=True) as out:

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for _ in tqdm(range(total_frames), desc=f"{video_path.name}", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    cropped = frame[top:bottom, left:right]
                    out.write(cropped)

            # Second pass: Resize to target resolution
            with video_capture(temp_cropped) as cap, \
                    video_writer(output_path, int(cap.get(cv2.CAP_PROP_FPS)),
                                 self.target_size, is_color=True) as out:

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                    out.write(resized)

            return output_path

        finally:
            if temp_cropped.exists():
                temp_cropped.unlink()
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
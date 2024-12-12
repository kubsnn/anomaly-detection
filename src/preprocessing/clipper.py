import cv2
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class VideoClipper:
    def __init__(self, clip_duration: int = 10):
        """
        Initialize the VideoClipper.

        Args:
            clip_duration (int): Duration of each clip in seconds
        """
        self.clip_duration = clip_duration

    def extract_clips(self, video_path: Path, output_dir: Path) -> List[Path]:
        """
        Extract fixed-duration clips from the video.

        Args:
            video_path (Path): Input video path
            output_dir (Path): Directory to save clips

        Returns:
            List[Path]: List of paths to generated clips
        """
        output_dir.mkdir(exist_ok=True)
        clip_paths = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_clip = fps * self.clip_duration

        for start_frame in range(0, total_frames, frames_per_clip):
            end_frame = min(start_frame + frames_per_clip, total_frames)
            if end_frame - start_frame < frames_per_clip / 2:  # Skip if clip too short
                continue

            clip_path = output_dir / f"{video_path.stem}_clip_{start_frame}_{end_frame}.mp4"

            if clip_path.exists():
                clip_paths.append(clip_path)
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clip_path), fourcc, fps,
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            frames_written = 0
            while frames_written < (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1

            out.release()
            clip_paths.append(clip_path)

        cap.release()
        return clip_paths
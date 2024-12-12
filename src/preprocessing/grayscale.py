import cv2
from pathlib import Path
import logging
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


class VideoConverter:
    def convert_to_grayscale(self, video_path: Path, output_path: Path) -> Path:
        """
        Convert video to grayscale while maintaining 3 channels.

        Args:
            video_path (Path): Input video path
            output_path (Path): Output video path

        Returns:
            Path: Path to the processed video
        """
        if output_path.exists():
            logger.info(f"Skipping existing grayscale video: {output_path}")
            return output_path

        with video_capture(video_path) as cap:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            with video_writer(output_path, fps, (width, height)) as out:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert to grayscale but maintain 3 channels
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    out.write(gray_3ch)

        return output_path
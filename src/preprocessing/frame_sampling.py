import cv2
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
import logging

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
def video_writer(path, fps, size, is_color=False):
    """Safely handle video writing with automatic closing"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size, isColor=is_color)
    try:
        if not writer.isOpened():
            raise IOError(f"Failed to create video writer: {path}")
        yield writer
    finally:
        writer.release()


def create_offset_samples(input_path: Path, output_dir: Path, version_number: int,
                          source_fps: int = 30, target_fps: int = 3,
                          target_size: tuple = None, grayscale: bool = False):
    """
    Create multiple sampled versions of a video, each starting from a different offset.

    Args:
        input_path (Path): Path to input video file
        output_dir (Path): Directory to save the processed videos
        version_number (int): Starting number for file versioning
        source_fps (int): Source video framerate (default: 30)
        target_fps (int): Desired output framerate (default: 3)
        target_size (tuple): Optional target resolution (width, height)
        grayscale (bool): Whether to convert the output to grayscale

    Returns:
        tuple: (number of videos created, list of output paths)
    """
    # Calculate sampling parameters
    sampling_interval = source_fps // target_fps
    num_offsets = sampling_interval

    logger.info(f"Creating {num_offsets} videos with sampling interval {sampling_interval}")
    logger.info(f"Source FPS: {source_fps}, Target FPS: {target_fps}")

    output_paths = []
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with video_capture(input_path) as cap:
        if not target_size:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_size = (frame_width, frame_height)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a video for each offset
        for i, offset in enumerate(range(num_offsets)):
            current_version = version_number + i
            output_path = output_dir / f"{input_path.stem}_{current_version}{input_path.suffix}"
            output_paths.append(output_path)

            with video_writer(output_path, target_fps, target_size,
                              is_color=not grayscale) as out:

                cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
                frame_count = offset
                processed_count = 0

                expected_frames = (total_frames - offset) // sampling_interval

                with tqdm(total=expected_frames,
                          desc=f"Processing version {current_version}",
                          leave=False) as pbar:

                    while frame_count < total_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if (frame_count - offset) % sampling_interval == 0:
                            if target_size != (frame.shape[1], frame.shape[0]):
                                frame = cv2.resize(frame, target_size,
                                                   interpolation=cv2.INTER_AREA)

                            if grayscale:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            out.write(frame)
                            processed_count += 1
                            pbar.update(1)

                        frame_count += 1

            logger.info(f"Created video {output_path.name} with {processed_count} frames")

    return num_offsets, output_paths

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
import sys

sys.path.append('..')
from utils.logger import setup_logging
from frame_sampling import video_capture, video_writer, create_offset_samples

logger = setup_logging(__name__)


# [detect_content_area function remains the same]

def process_video(input_path: Path, output_dir: Path, version_start: int,
                  target_fps: int = 3, target_size: tuple = (320, 180)):
    """
    Process a video by detecting content area, cropping, and creating multiple sampled versions.

    Args:
        input_path (Path): Path to input video
        output_dir (Path): Directory to save processed versions
        version_start (int): Starting number for version counting
        target_fps (int): Target frames per second for output
        target_size (tuple): Target resolution (width, height) for output

    Returns:
        int: Number of versions created
    """
    # First detect content area
    with video_capture(input_path) as cap:
        # Sample frames uniformly across the video
        sample_frames = 5
        frame_positions = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, sample_frames, dtype=int)

        lefts, rights, tops, bottoms = [], [], [], []

        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue

            coords = detect_content_area(frame)
            if coords:
                left, right, top, bottom = coords
                lefts.append(left)
                rights.append(right)
                tops.append(top)
                bottoms.append(bottom)

        if not lefts:
            raise ValueError(f"Could not detect content area in {input_path}")

        # Use median values for stable cropping
        left = int(np.median(lefts))
        right = int(np.median(rights))
        top = int(np.median(tops))
        bottom = int(np.median(bottoms))

    # Create temporary directory for intermediate files
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_cropped = temp_dir / f"crop_{input_path.name}"

    try:
        # First pass: Crop the video
        with video_capture(input_path) as cap, \
                video_writer(temp_cropped, int(cap.get(cv2.CAP_PROP_FPS)),
                             (right - left, bottom - top), is_color=True) as out:

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(range(total_frames), desc=f"Cropping {input_path.name}", leave=False):
                ret, frame = cap.read()
                if not ret:
                    break

                cropped = frame[top:bottom, left:right]
                out.write(cropped)

        # Second pass: Create sampled versions
        num_versions, output_paths = create_offset_samples(
            temp_cropped,
            output_dir,
            version_number=version_start,
            source_fps=30,
            target_fps=target_fps,
            target_size=target_size,
            grayscale=True
        )

        return num_versions

    finally:
        # Clean up temporary files
        if temp_cropped.exists():
            temp_cropped.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                logger.warning(f"Could not remove temp directory: {temp_dir}")


def process_dataset(input_dir: str, output_dir: str):
    """
    Process all videos in the input directory and save processed versions to the output directory.

    Args:
        input_dir (str): Directory containing source videos
        output_dir (str): Directory to save processed videos
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize counters
    total_source_processed = 0
    total_versions_created = 0
    total_errors = 0
    current_version = 0

    # Find all video files
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    video_files = [f for f in video_files if f.is_file()]

    logger.info(f"Found {len(video_files)} videos to process in {input_dir}")

    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Check for existing processed versions
            existing_versions = list(output_dir.glob(f"processed_{video_path.stem}_*{video_path.suffix}"))
            if existing_versions:
                logger.debug(f"Skipping {video_path.name}: already processed")
                continue

            # Process the video
            num_versions = process_video(video_path, output_dir,
                                         version_start=current_version)

            current_version += num_versions
            total_source_processed += 1
            total_versions_created += num_versions

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {str(e)}")
            total_errors += 1
            continue

    logger.info(f"\nProcessing completed for {input_dir}:")
    logger.info(f"Successfully processed: {total_source_processed} source videos")
    logger.info(f"Total versions created: {total_versions_created}")
    logger.info(f"Failed to process: {total_errors} videos")
    logger.info(f"Final version number: {current_version - 1}")


if __name__ == "__main__":
    logger.warning("This script should be run through the pipeline, not directly.")
    logger.warning("Run 'python run_pipeline.py' instead.")
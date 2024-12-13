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


def detect_content_area(frame, threshold=30):
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

    for x in range(w // 2):
        if is_content_col(gray[:, x], threshold):
            left = x
            break

    for x in range(w - 1, w // 2, -1):
        if is_content_col(gray[:, x], threshold):
            right = x
            break

    for y in range(h // 2):
        if is_content_row(gray[y, :], threshold):
            top = y
            break

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


def process_video(input_path: Path, output_dir: Path, version_start: int,
                  target_fps: int = 3, target_size: tuple = (320, 180)):
    """
    Process a video by detecting content area, cropping, and creating multiple sampled versions.
    """
    # First detect content area
    with video_capture(input_path) as cap:
        content_coords = None
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

        left = int(np.median(lefts))
        right = int(np.median(rights))
        top = int(np.median(tops))
        bottom = int(np.median(bottoms))

    # Create temporary file for cropped video
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_cropped = temp_dir / f"temp_cropped_{input_path.name}"

    try:
        # First pass: Crop the video
        with video_capture(input_path) as cap, \
                video_writer(temp_cropped, int(cap.get(cv2.CAP_PROP_FPS)),
                             (right - left, bottom - top), is_color=True) as out:

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(range(total_frames), desc=f"{input_path.name}", leave=False):
                ret, frame = cap.read()
                if not ret:
                    break

                cropped = frame[top:bottom, left:right]
                out.write(cropped)

        # Second pass: Create multiple sampled versions
        num_versions, output_paths = create_offset_samples(
            temp_cropped,
            output_dir,
            version_number=version_start,
            source_fps=30,  # Assuming source is 30fps
            target_fps=target_fps,
            target_size=target_size,
            grayscale=True
        )

        # logger.info(f"Processed {input_path.name} into {num_versions} versions")
        # logger.info(f"Crop coordinates: ({left}, {right}, {top}, {bottom})")

        return num_versions

    finally:
        if temp_cropped.exists():
            temp_cropped.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()


def process_dataset(input_dir: str, output_dir: str):
    """
    Process all videos in the input directory and save processed versions to the output directory.

    Args:
        input_dir (str): Directory containing source videos
        output_dir (str): Directory to save processed videos
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    total_source_processed = 0
    total_versions_created = 0
    total_errors = 0
    current_version = 0

    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    video_files = [f for f in video_files if f.is_file()]

    logger.info(f"Found {len(video_files)} videos to process")

    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Check if any version of this video already exists
            existing_versions = list(output_dir.glob(f"temp_cropped_{video_path.stem}_*{video_path.suffix}"))
            if existing_versions:
                #logger.info(f"Skipping existing video versions: {video_path.stem}")
                continue

            num_versions = process_video(video_path, output_dir,
                                         version_start=current_version)

            current_version += num_versions
            total_source_processed += 1
            total_versions_created += num_versions

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {str(e)}")
            total_errors += 1
            continue

    logger.info(f"\nProcessing completed:")
    logger.info(f"Successfully processed: {total_source_processed} source videos")
    logger.info(f"Total versions created: {total_versions_created}")
    logger.info(f"Failed to process: {total_errors} videos")
    logger.info(f"Final version number: {current_version - 1}")


def main():
    input_dir = "../../data/UBI_FIGHTS/videos/normal"
    output_dir = "../../data/UBI_FIGHTS/videos/normal/processed"

    try:
        logger.info("Starting video processing pipeline...")
        process_dataset(input_dir, output_dir)
        logger.info("Video processing completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

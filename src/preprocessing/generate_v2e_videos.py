import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import subprocess, sys
sys.path.append('..')
from pathlib import Path
import cv2
from tqdm import tqdm
from utils import setup_logging

logger = setup_logging(__name__)

def get_video_resolution(video_path: Path):
    """
    Get the resolution (width, height) of a video.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        tuple: (width, height) of the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return width, height


def convert_to_dvs(input_video: Path, output_dir: Path, disable_slomo: bool = True):
    """
    Convert a video to DVS events using the v2e tool.

    Args:
        input_video (Path): Path to the input video.
        output_dir (Path): Directory where the output DVS video will be saved.
        disable_slomo (bool): Whether to disable SloMo interpolation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dvs_output_file = output_dir / f"{input_video.stem}_dvs.avi"

    if dvs_output_file.exists():
        logger.info(f"DVS video already exists: {dvs_output_file}")
        return

    # Get resolution of the input video
    try:
        width, height = get_video_resolution(input_video)
    except Exception as e:
        logger.error(f"Could not determine resolution of {input_video}: {e}")
        return

    # Prepare v2e command
    command = [
        "v2e",
        "--input", str(input_video),
        "--output_folder", str(output_dir),
        "--output_width", str(width),
        "--output_height", str(height),
        "--disable_slomo",
        "--batch_size", "768",
        "--avi_frame_rate", "3",
        "--dvs_exposure", "duration", "0.333",
        "--no_preview",
    ]

    logger.info(f"Converting {input_video} to DVS with resolution {width}x{height}...")
    try:
        subprocess.run(command, check=True)
        logger.info(f"Generated DVS video: {dvs_output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {input_video}: {str(e)}")


def process_directory(base_path: Path, category: str, split: str):
    """
    Process a specific directory (normal or fight) for a given split.

    Args:
        base_path (Path): Base path to the dataset.
        category (str): The category ("normal" or "fight").
        split (str): The data split ("train", "val", "test").
    """
    processed_dir = base_path / "videos" / category / split / "processed"
    dvs_output_dir = processed_dir / "v2e"

    if not processed_dir.exists():
        logger.warning(f"Processed directory not found: {processed_dir}")
        return 0, 0, 0

    # Find all processed videos
    video_files = list(processed_dir.glob("*.mp4")) + list(processed_dir.glob("*.avi"))
    video_files = [f for f in video_files if f.is_file()]

    if not video_files:
        logger.warning(f"No videos found in {processed_dir}")
        return 0, 0, 0

    logger.info(f"Processing '{category}' category for {split} split: {len(video_files)} videos")

    total_processed, total_skipped, total_errors = 0, 0, 0

    for video_path in tqdm(video_files, desc=f"{category}-{split} v2e conversion"):
        try:
            dvs_video_dir = dvs_output_dir / video_path.stem

            # Skip if already processed
            if dvs_video_dir.exists():
                logger.debug(f"Skipping existing video: {video_path}")
                total_skipped += 1
                continue

            # Convert to DVS events
            convert_to_dvs(video_path, dvs_video_dir)
            total_processed += 1

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {str(e)}")
            total_errors += 1

    return total_processed, total_skipped, total_errors


def apply_v2e_to_processed_videos(base_path: str):
    """
    Apply v2e to all videos in the `processed/` directories of the dataset.

    Args:
        base_path (str): Base directory containing the dataset.
    """
    base_path = Path(base_path)
    splits = ["train", "val", "test"]
    categories = ["normal", "fight"]

    total_processed, total_skipped, total_errors = 0, 0, 0

    for category in categories:
        for split in splits:
            processed, skipped, errors = process_directory(base_path, category, split)
            total_processed += processed
            total_skipped += skipped
            total_errors += errors

    logger.info("\n=== v2e Conversion Summary ===")
    logger.info(f"Successfully processed: {total_processed} videos")
    logger.info(f"Skipped already processed: {total_skipped} videos")
    logger.info(f"Failed to process: {total_errors} videos")


if __name__ == "__main__":
    # Set the base path for the dataset
    base_path = "../../data/UBI_FIGHTS"

    try:
        logger.info("Starting v2e conversion pipeline...")
        apply_v2e_to_processed_videos(base_path)
        logger.info("v2e conversion pipeline completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

sys.path.append('..')
from utils import setup_logging
from .crop import VideoCropper
from .clipper import VideoClipper
from .grayscale import VideoConverter
from .generate_v2e_videos import apply_v2e_to_processed_videos

logger = setup_logging(__name__)


class PreprocessingPipeline:
    def __init__(self, base_path: str, target_size: tuple = (320, 180)):
        """
        Initialize the preprocessing pipeline.

        Args:
            base_path (str): Base path to UBI_FIGHTS dataset
            target_size (tuple): Target resolution (width, height) for videos
        """
        self.base_path = Path(base_path)
        self.videos_dir = self.base_path / "videos"
        self.output_base = self.base_path / "processed"
        self.temp_dir = self.output_base / "temp"
        self.target_size = target_size

        # Create processing objects
        self.cropper = VideoCropper(target_size=target_size)
        self.clipper = VideoClipper(clip_duration=10)
        self.converter = VideoConverter()

        # Create necessary directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_base.mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path: Path, category: str):
        """Process a single video through the entire pipeline."""
        try:
            logger.info(f"Processing {category} video: {video_path}")

            # Create temporary directories for each stage
            category_dir = self.temp_dir / category
            temp_crop_dir = category_dir / "cropped"
            temp_clips_dir = category_dir / "clips"
            temp_gray_dir = category_dir / "gray"

            for dir_path in [temp_crop_dir, temp_clips_dir, temp_gray_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Crop and resize video
            cropped_path = temp_crop_dir / f"cropped_{video_path.name}"
            cropped_video = self.cropper.crop_video(video_path, cropped_path)

            # Step 2: Extract clips
            clips = self.clipper.extract_clips(cropped_video, temp_clips_dir)

            # Step 3: Convert each clip to grayscale
            processed_clips = []
            for clip in clips:
                gray_path = temp_gray_dir / f"gray_{clip.name}"
                gray_video = self.converter.convert_to_grayscale(clip, gray_path)
                if gray_video and gray_video.exists():
                    processed_clips.append(gray_video)

            return processed_clips

        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            raise

    def process_category(self, category: str):
        """Process all videos in a specific category (normal/fight)."""
        category_dir = self.videos_dir / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return

        # Get all videos, including those in subdirectories
        video_files = []
        for ext in ('*.mp4', '*.avi'):
            video_files.extend(category_dir.rglob(ext))
        video_files = [f for f in video_files if f.is_file()]

        if not video_files:
            logger.warning(f"No video files found in {category_dir}")
            return

        logger.info(f"Processing {category} videos: {len(video_files)} files found")

        for video_path in tqdm(video_files, desc=f"Processing {category} videos"):
            try:
                self.process_video(video_path, category)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {str(e)}")
                continue

    def process_dataset(self):
        """Process all videos in both normal and fight categories."""
        categories = ["normal", "fight"]

        try:
            # Step 1-3: Process all videos through cropping, clipping, and grayscale conversion
            for category in categories:
                self.process_category(category)

            # Step 4: Apply v2e to all processed videos using existing implementation
            logger.info("Applying v2e transformation to processed videos...")
            apply_v2e_to_processed_videos(self.base_path)

        finally:
            # Cleanup temporary files
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary files")


def main():
    # Set the base path
    base_path = "../../data/UBI_FIGHTS"

    # Create and run the pipeline
    try:
        pipeline = PreprocessingPipeline(
            base_path=base_path,
            target_size=(320, 180)  # 16:9 aspect ratio at reduced resolution
        )

        logger.info("Starting video preprocessing pipeline...")
        pipeline.process_dataset()
        logger.info("Video preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
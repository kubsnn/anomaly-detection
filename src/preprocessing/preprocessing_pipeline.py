import os
import sys
from pathlib import Path
import logging
import time

sys.path.append('..')

from process_videos import process_dataset
from generate_v2e_videos import apply_v2e_to_processed_videos
from copy_v2e_output import copy_and_convert_videos, prompt_to_remove_empty_dirs
from utils.logger import setup_logging

logger = setup_logging(__name__)


class Pipeline:
    def __init__(self, base_path: str):
        """
        Initialize the video processing pipeline.

        Args:
            base_path (str): Base directory containing the dataset
        """
        self.base_path = Path(base_path)
        self.videos_path = self.base_path / "videos"
        self.fight_path = self.videos_path / "fight"
        self.processed_path = self.fight_path / "processed"

        # Ensure directories exist
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run the complete pipeline."""
        start_time = time.time()

        try:
            logger.info("Starting video processing pipeline...")

            # Step 1: Process videos (detect content area and crop)
            logger.info("\n=== Step 1: Processing Videos ===")
            process_dataset(
                input_dir=str(self.fight_path),
                output_dir=str(self.processed_path)
            )

            # Step 2: Frame sampling is handled within process_dataset
            logger.info("\n=== Step 2: Frame Sampling ===")
            logger.info("Frame sampling is integrated into the video processing step")

            # Step 3: Apply v2e conversion
            logger.info("\n=== Step 3: V2E Conversion ===")
            apply_v2e_to_processed_videos(str(self.base_path))

            # Step 4: Copy and convert videos to final structure
            logger.info("\n=== Step 4: Copy and Convert Videos ===")
            empty_dirs = copy_and_convert_videos(str(self.base_path), logger)
            prompt_to_remove_empty_dirs(empty_dirs, logger)

            # Calculate total processing time
            total_time = time.time() - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)

            logger.info("\n=== Pipeline Complete ===")
            logger.info(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        except KeyboardInterrupt:
            logger.warning("\nPipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    # Set the base path for the dataset
    base_path = "../../data/UBI_FIGHTS"

    # Create and run the pipeline
    pipeline = Pipeline(base_path)
    pipeline.run()


if __name__ == "__main__":
    main()
import sys
sys.path.append('..')
from pathlib import Path
import time
from utils.logger import setup_logging
from process_videos import process_dataset
from cut_fights_from_videos import extract_fight_clips
from split_videos import split_processed_videos
from generate_v2e_videos import apply_v2e_to_processed_videos
from copy_v2e_output import copy_and_convert_videos, prompt_to_remove_empty_dirs
from set_creator import create_dataset_splits

logger = setup_logging(__name__)

class Pipeline:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.start_time = None

    def _log_step_start(self, step_name: str):
        logger.info("\n" + "=" * 50)
        logger.info(f"Starting Step: {step_name}")
        logger.info("=" * 50)
        return time.time()

    def _log_step_complete(self, step_name: str, start_time: float):
        duration = time.time() - start_time
        logger.info("\n" + "-" * 50)
        logger.info(f"Completed Step: {step_name}")
        logger.info(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        logger.info("-" * 50 + "\n")

    def process_normal_videos(self):
        """Step 1: Process normal videos"""
        start = self._log_step_start("Processing Normal Videos")
        process_dataset(
            str(self.base_path / "videos/normal"),
            str(self.base_path / "videos/normal/processed")
        )
        self._log_step_complete("Processing Normal Videos", start)

    def extract_fight_segments(self):
        """Step 2: Extract fight segments from videos"""
        start = self._log_step_start("Extracting Fight Segments")
        extract_fight_clips(self.base_path, min_output_duration=3.0)
        self._log_step_complete("Extracting Fight Segments", start)

    def process_fight_videos(self):
        """Step 3: Process extracted fight videos"""
        start = self._log_step_start("Processing Fight Videos")
        process_dataset(
            str(self.base_path / "videos/fight/cut_fights"),
            str(self.base_path / "videos/fight/cut_fights/processed")
        )
        self._log_step_complete("Processing Fight Videos", start)

    def generate_dvs_videos(self):
        """Step 4: Generate DVS videos using v2e"""
        start = self._log_step_start("Generating DVS Videos")
        apply_v2e_to_processed_videos(str(self.base_path))
        self._log_step_complete("Generating DVS Videos", start)

    def copy_v2e_results(self):
        """Step 5: Copy and organize v2e output"""
        start = self._log_step_start("Copying V2E Results")
        empty_dirs = copy_and_convert_videos(str(self.base_path), logger)
        prompt_to_remove_empty_dirs(empty_dirs, logger)
        self._log_step_complete("Copying V2E Results", start)

    def split_videos(self, segment_length: int = 10):
        """Step 6: Split processed videos into smaller segments"""
        start = self._log_step_start("Splitting Processed Videos")
        split_processed_videos(self.base_path, segment_length)
        self._log_step_complete("Splitting Processed Videos", start)

    def create_datasets(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Step 7: Create train, validation, and test datasets"""
        start = self._log_step_start("Creating Dataset Splits")
        create_dataset_splits(self.base_path, train_ratio, val_ratio)
        self._log_step_complete("Creating Dataset Splits", start)

    def run(self, segment_length: int = 10, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Run the complete pipeline"""
        self.start_time = time.time()
        logger.info("\n" + "#" * 60)
        logger.info("Starting Video Processing Pipeline")
        logger.info("#" * 60 + "\n")

        try:
            # Step 1: Process normal videos
            self.process_normal_videos()

            # Step 2: Extract fight segments and filter short videos
            self.extract_fight_segments()

            # Step 3: Process extracted fight videos
            self.process_fight_videos()

            # Step 4: Generate DVS videos
            self.generate_dvs_videos()

            # Step 5: Copy and organize v2e output
            self.copy_v2e_results()

            # Step 6: Split videos into segments
            self.split_videos(segment_length)

            # Step 7: Create dataset splits from segmented videos
            self.create_datasets(train_ratio, val_ratio)

            total_duration = time.time() - self.start_time
            logger.info("\n" + "#" * 60)
            logger.info("Pipeline Completed Successfully!")
            logger.info(f"Total Duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
            logger.info("#" * 60 + "\n")

        except KeyboardInterrupt:
            logger.warning("\nPipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\nPipeline failed: {str(e)}")
            raise

def main():
    base_path = "../../data/UBI_FIGHTS"

    pipeline = Pipeline(base_path)
    pipeline.run(
        segment_length=10,  
        train_ratio=0.7,   
        val_ratio=0.15    
    )

if __name__ == "__main__":
    main()

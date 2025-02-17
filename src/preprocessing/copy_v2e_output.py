import os, sys
sys.path.append('..')

import subprocess
from pathlib import Path
import shutil

from utils import setup_logging



def copy_and_convert_videos(base_path: str, logger):
    """
    Copy and convert videos from processed\v2e subdirectories to the target v2e structure.
    Converts .avi files to .mp4 in the target location, removes '_processed' from names, and skips if already converted.
    Tracks empty directories for potential removal at the end of the pipeline.

    Args:
        base_path (str): Base directory containing the `UBI_FIGHTS` dataset.
    """
    base_path = Path(base_path)
    processed_base = base_path / "videos" / "normal"
    target_base = base_path / "v2e" / "videos" / "normal"

    splits = ["train", "test", "val"]
    empty_dirs = []  

    for split in splits:
        processed_split_dir = processed_base / split / "processed" / "v2e"
        target_split_dir = target_base / split

        if not processed_split_dir.exists():
            logger.warning(f"Processed directory does not exist: {processed_split_dir}")
            continue

        target_split_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in processed_split_dir.iterdir():
            if not subdir.is_dir():
                continue

            dvs_video_file = subdir / "dvs-video.avi"
            if not dvs_video_file.exists():
                logger.warning(f"Video not found: {dvs_video_file}")
                empty_dirs.append(subdir)
                continue

            clean_name = subdir.name.replace("_processed", "").replace("temp_cropped_", "").replace("cut_", "")
            target_file = target_split_dir / f"{clean_name}.mp4"

            if target_file.exists():
                logger.info(f"Skipping: {target_file} already exists.")
                continue

            try:
                rel_dvs_video_file = os.path.relpath(dvs_video_file, base_path)
                rel_target_file = os.path.relpath(target_file, base_path)
                logger.info(f"Converting and copying: ...{rel_dvs_video_file} -> ...{rel_target_file}")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i", str(dvs_video_file),
                        "-c:v", "libx264",
                        "-crf", "23",
                        "-preset", "medium",
                        "-c:a", "aac",
                        str(target_file)
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info(f"OK!")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to convert {dvs_video_file}: {e}")

        logger.info(f"Completed copying and converting videos for {split} split.")

    return empty_dirs


def prompt_to_remove_empty_dirs(empty_dirs, logger):
    """
    Prompt the user to remove empty directories tracked during the process.

    Args:
        empty_dirs (list): List of empty directory paths to potentially remove.
    """
    if not empty_dirs:
        logger.info("No empty directories to remove.")
        return

    logger.warning("\nThe following directories are empty or contain only residual files:")
    for dir_path in empty_dirs:
        logger.warning(f"- {dir_path}")

    remove = input("Do you want to forcefully remove these directories? (y/n): ").strip().lower()
    if remove == 'y':
        for dir_path in empty_dirs:
            try:
                shutil.rmtree(dir_path)  
                logger.info(f"Removed: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to remove {dir_path}: {e}")
    else:
        logger.info("Empty directories were not removed.")


if __name__ == "__main__":
    logger = setup_logging(__name__)

    base_path = "../../data/UBI_FIGHTS"

    try:
        logger.info("Starting video copy and convert pipeline...")
        empty_dirs = copy_and_convert_videos(base_path, logger)
        prompt_to_remove_empty_dirs(empty_dirs, logger)
        logger.info("Video copy and convert pipeline completed successfully!")
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")

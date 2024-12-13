import sys

sys.path.append('..')
from pathlib import Path
import shutil
import random
from tqdm import tqdm
from utils.logger import setup_logging

logger = setup_logging(__name__)


def create_dataset_splits(base_path: Path, train_ratio=0.7, val_ratio=0.15):
    """
    Create train, validation, and test datasets.
    - Train and validation contain only normal videos
    - Test set has equal numbers of normal and fight videos

    Args:
        base_path (Path): Base directory containing the dataset
        train_ratio (float): Proportion of normal videos for training
        val_ratio (float): Proportion of normal videos for validation
        (remaining normal videos will be used for testing)
    """
    # Setup paths
    normal_dir = base_path / "videos" / "normal" / "processed" / "split"
    fight_dir = base_path / "videos" / "fight" / "cut_fights" / "processed" / "split"

    dataset_dir = base_path / "dataset"
    splits = ['train', 'val', 'test']
    categories = ['normal', 'fight']

    # Create directory structure
    for split in splits:
        for category in categories:
            (dataset_dir / split / category).mkdir(parents=True, exist_ok=True)

    # Collect all videos
    normal_videos = list(normal_dir.glob("*.mp4")) + list(normal_dir.glob("*.avi"))
    fight_videos = list(fight_dir.glob("*.mp4")) + list(fight_dir.glob("*.avi"))

    if not normal_videos:
        raise ValueError(f"No normal videos found in {normal_dir}")
    if not fight_videos:
        raise ValueError(f"No fight videos found in {fight_dir}")

    logger.info(f"Found {len(normal_videos)} normal videos and {len(fight_videos)} fight videos")

    # Randomly shuffle videos
    random.shuffle(normal_videos)
    random.shuffle(fight_videos)

    # Calculate splits for normal videos
    total_normal = len(normal_videos)
    train_size = int(total_normal * train_ratio)
    val_size = int(total_normal * val_ratio)

    # Number of videos for test set (equal number of fight and normal)
    test_size = min(len(fight_videos), total_normal - train_size - val_size)

    # Split normal videos
    normal_splits = {
        'train': normal_videos[:train_size],
        'val': normal_videos[train_size:train_size + val_size],
        'test': normal_videos[train_size + val_size:train_size + val_size + test_size]
    }

    # Use equal number of fight videos for test set
    fight_splits = {
        'train': [],
        'val': [],
        'test': fight_videos[:test_size]
    }

    # Copy videos to their respective directories
    total_copied = 0

    for split in splits:
        logger.info(f"\nProcessing {split} split:")

        # Copy normal videos
        for video_path in tqdm(normal_splits[split], desc=f"Copying normal videos to {split}"):
            dest_path = dataset_dir / split / "normal" / video_path.name
            shutil.copy2(video_path, dest_path)
            total_copied += 1

        # Copy fight videos (test set only)
        for video_path in tqdm(fight_splits[split], desc=f"Copying fight videos to {split}"):
            dest_path = dataset_dir / split / "fight" / video_path.name
            shutil.copy2(video_path, dest_path)
            total_copied += 1

    # Log dataset statistics
    logger.info("\nDataset creation completed:")
    logger.info(f"Train set: {len(normal_splits['train'])} normal videos")
    logger.info(f"Validation set: {len(normal_splits['val'])} normal videos")
    logger.info(f"Test set: {len(normal_splits['test'])} normal videos, "
                f"{len(fight_splits['test'])} fight videos")
    logger.info(f"Total videos copied: {total_copied}")


def main():
    base_path = Path("../../data/UBI_FIGHTS")

    try:
        logger.info("Starting dataset creation...")
        create_dataset_splits(base_path)
        logger.info("Dataset creation completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
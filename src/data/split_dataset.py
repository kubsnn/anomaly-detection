import shutil
import random
from pathlib import Path
import logging


def setup_dataset_split(base_path: str, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("Ratios must sum to 1")

    normal_dir = Path(base_path) / 'videos' / 'normal'
    if not normal_dir.exists():
        raise FileNotFoundError(f"Directory not found: {normal_dir}")

    split_dirs = {
        'train': normal_dir / 'train',
        'val': normal_dir / 'val',
        'test': normal_dir / 'test'
    }

    for dir_path in split_dirs.values():
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    video_files = [f for f in normal_dir.glob('*') if f.suffix in ('.mp4', '.avi') and f.is_file()]

    if not video_files:
        raise ValueError(f"No video files found in {normal_dir}")

    random.seed(42)  # reproducibility
    random.shuffle(video_files)

    total_videos = len(video_files)
    train_size = int(total_videos * train_ratio)
    val_size = int(total_videos * val_ratio)

    train_files = video_files[:train_size]
    val_files = video_files[train_size:train_size + val_size]
    test_files = video_files[train_size + val_size:]

    for files, split_name in [(train_files, 'train'),
                              (val_files, 'val'),
                              (test_files, 'test')]:
        for file_path in files:
            dest_path = split_dirs[split_name] / file_path.name
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"Moved {file_path.name} to {split_name}")

    logger.info("\nDataset Split Summary:")
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Training set: {len(train_files)} videos ({train_ratio * 100:.1f}%)")
    logger.info(f"Validation set: {len(val_files)} videos ({val_ratio * 100:.1f}%)")
    logger.info(f"Test set: {len(test_files)} videos ({test_ratio * 100:.1f}%)")


if __name__ == "__main__":
    base_path = "../../../UBI_FIGHTS"
    try:
        setup_dataset_split(base_path)
        print("Dataset split completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
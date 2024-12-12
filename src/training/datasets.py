from pathlib import Path
import random
from torch.utils.data import DataLoader, random_split
from data.clipper import ProcessedVideoDataset


def create_dataloaders(processed_dir: str, config: dict):
    """
    Create train, validation, and test dataloaders from preprocessed videos.

    Args:
        processed_dir (str): Directory containing preprocessed videos
        config (dict): Configuration dictionary containing:
            - batch_size: Batch size for dataloaders
            - target_size: Tuple of (height, width) for frames
            - train_ratio: Fraction of data for training (default: 0.7)
            - val_ratio: Fraction of data for validation (default: 0.2)
            - test_ratio: Fraction of data for testing (default: 0.1)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get all processed video paths
    processed_dir = Path(processed_dir)
    video_paths = []
    for ext in ('*.mp4', '*.avi'):
        video_paths.extend([str(p) for p in processed_dir.rglob(ext)])

    # Set random seed for reproducibility
    random.seed(42)
    random.shuffle(video_paths)

    # Get split ratios from config or use defaults
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.2)
    test_ratio = config.get('test_ratio', 0.1)

    # Calculate split sizes
    total_size = len(video_paths)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split paths
    train_paths = video_paths[:train_size]
    val_paths = video_paths[train_size:train_size + val_size]
    test_paths = video_paths[train_size + val_size:]

    # Create datasets
    train_dataset = ProcessedVideoDataset(train_paths, target_size=config['target_size'])
    val_dataset = ProcessedVideoDataset(val_paths, target_size=config['target_size'])
    test_dataset = ProcessedVideoDataset(test_paths, target_size=config['target_size'])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
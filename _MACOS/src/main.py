import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import platform
import sys

# Import from main src directory
from src.data.clipper import VideoClipDataset
from src.utils.training import pretrain_autoencoder
from models.autoencoder import MPSVideoAutoencoder as VideoAutoencoder  # Using alias

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def verify_macos():
    """Verify that we're running on macOS."""
    if platform.system() != 'Darwin':
        raise RuntimeError("This script is intended to run only on macOS systems.")
    logger.info(f"Running on macOS version: {platform.mac_ver()[0]}")


def get_mps_device():
    """Get MPS device for Apple Silicon or fall back to CPU."""
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.warning("MPS not available because PyTorch is not built with MPS support.")
        else:
            logger.warning("MPS not available even though PyTorch has MPS support.")
        device = torch.device("cpu")
        logger.warning("Using CPU instead.")
        return device

    device = torch.device("mps")
    logger.info("Using Apple Silicon GPU (MPS)")

    torch.set_default_dtype(torch.float32)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS device: {device}")

    return device


def get_video_paths(base_path, subset_size=5):
    """
    Get paths for a subset of videos from both fight and normal directories.

    Args:
        base_path (str): Base path to the video directory
        subset_size (int): Number of videos to use from each category

    Returns:
        list: List of video file paths
    """
    # Verify base path exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    # Get fight videos
    fight_dir = os.path.join(base_path, 'videos', 'fight')
    if not os.path.exists(fight_dir):
        raise FileNotFoundError(f"Fight videos directory not found: {fight_dir}")

    fight_videos = []
    for f in sorted(os.listdir(fight_dir)):
        if f.endswith(('.mp4', '.avi')):
            path = os.path.join(fight_dir, f)
            if os.path.isfile(path):
                fight_videos.append(path)
    fight_videos = fight_videos[:subset_size]

    # Get normal videos
    normal_dir = os.path.join(base_path, 'videos', 'normal')
    if not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Normal videos directory not found: {normal_dir}")

    normal_videos = []
    for f in sorted(os.listdir(normal_dir)):
        if f.endswith(('.mp4', '.avi')):
            path = os.path.join(normal_dir, f)
            if os.path.isfile(path):
                normal_videos.append(path)
    normal_videos = normal_videos[:subset_size]

    # Combine and verify
    video_paths = fight_videos + normal_videos

    if not video_paths:
        raise ValueError("No valid video files found")

    logger.info(f"Found {len(fight_videos)} fight videos and {len(normal_videos)} normal videos")
    logger.info(f"Total videos for training: {len(video_paths)}")

    return video_paths


def setup_training_environment(config):
    """Setup the training environment with appropriate settings for MPS."""
    device = get_mps_device()

    torch.manual_seed(42)

    config['device'] = device

    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    return config


def main():
    # Verify we're on macOS
    verify_macos()

    # Basic configuration
    CONFIG = {
        'base_path': '/Users/morpheus/Documents/PolitechnikaPoznanska/Semestr7/WykrywanieAnomaliiWZachowaniachSpolecznych/UBI_FIGHTS',
        'subset_size': 5,
        'batch_size': 16,  # Can be adjusted based on available memory
        'num_epochs': 2,
        'learning_rate': 1e-4,
        'clip_length': 16,
        'input_channels': 3,
        'latent_dim': 256,
        'target_size': (64, 64)
    }

    # Setup training environment and get updated config
    CONFIG = setup_training_environment(CONFIG)

    try:
        # Get video paths
        video_paths = get_video_paths(
            CONFIG['base_path'],
            CONFIG['subset_size']
        )

        # Create dataset
        logger.info("Creating video dataset...")
        dataset = VideoClipDataset(
            video_paths=video_paths,
            clip_length=CONFIG['clip_length'],
            clip_overlap=0.5,
            min_clips=1,
            augment=True,
            target_size=CONFIG['target_size']
        )

        # Create data loader optimized for MPS
        loader = DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=2,  # Reduced for macOS compatibility
            pin_memory=True if CONFIG['device'].type == "mps" else False,
            persistent_workers=True  # Keep workers alive between iterations
        )

        logger.info(f"Created dataloader with {len(dataset)} clips")

        # Initialize model
        logger.info("Initializing model...")
        model = VideoAutoencoder(
            input_channels=CONFIG['input_channels'],
            latent_dim=CONFIG['latent_dim']
        ).to(CONFIG['device'])

        # Training loop
        logger.info("Starting training...")
        try:
            history = pretrain_autoencoder(
                model=model,
                train_loader=loader,
                num_epochs=CONFIG['num_epochs'],
                learning_rate=CONFIG['learning_rate'],
                device=CONFIG['device']
            )
            logger.info("Training completed successfully!")

            # Save the model
            save_dir = Path('_MACOS/models')
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / 'autoencoder_mps.pth'
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

            # Save training history
            import json
            history_path = save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f)
            logger.info(f"Training history saved to {history_path}")

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Setup failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}")
        raise
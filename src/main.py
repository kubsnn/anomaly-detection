import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import platform

from data.clipper import VideoClipDataset
from models.autoencoder import VideoAutoencoder
from utils.training import pretrain_autoencoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_device():
    """Get the appropriate device for training."""
    device = torch.device("cpu")
    logger.info("Using CPU for computations")
    return device



def get_video_paths(base_path, subset_size=5):
    """
    Get paths for a subset of videos from both fight and normal directories.
    """
    fight_dir = os.path.join(base_path, 'videos', 'fight')
    fight_videos = sorted([os.path.join(fight_dir, f) for f in os.listdir(fight_dir)
                           if f.endswith(('.mp4', '.avi'))])[:subset_size]

    normal_dir = os.path.join(base_path, 'videos', 'normal')
    normal_videos = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                            if f.endswith(('.mp4', '.avi'))])[:subset_size]

    annotation_dir = os.path.join(base_path, 'annotation')
    annotation_paths = []
    for video_path in fight_videos:
        video_name = os.path.basename(video_path)
        annotation_name = f"{os.path.splitext(video_name)[0]}.csv"
        annotation_path = os.path.join(annotation_dir, annotation_name)
        if os.path.exists(annotation_path):
            annotation_paths.append(annotation_path)
        else:
            logger.warning(f"Annotation not found for {video_name}")

    video_paths = fight_videos + normal_videos

    return video_paths, annotation_paths


def main():
    device = get_device()
    logger.info(f"Using device: {device}")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    elif device.type == "mps":
        pass

    CONFIG = {
        'base_path': '/Users/morpheus/Documents/PolitechnikaPoznanska/Semestr7/WykrywanieAnomaliiWZachowaniachSpolecznych/UBI_FIGHTS',
        'subset_size': 10,
        'batch_size': 4,
        'num_epochs': 4,
        'learning_rate': 1e-4,
        'clip_length': 16,
        'input_channels': 3,
        'latent_dim': 256,
        'target_size': (64, 64),
        'device': get_device()
    }


    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True

    logger.info(f"Using device: {CONFIG['device']}")

    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if device.type == "mps":
        logger.info("Using Apple Silicon Neural Engine")

    logger.info("Loading video paths...")

    video_paths, annotation_paths = get_video_paths(
        CONFIG['base_path'],
        CONFIG['subset_size']
    )

    logger.info(f"Found {len(video_paths)} videos for testing")

    logger.info("Creating video dataset...")
    dataset = VideoClipDataset(
        video_paths=video_paths,
        clip_length=CONFIG['clip_length'],
        clip_overlap=0.5,
        min_clips=1,
        augment=True,
        target_size=CONFIG['target_size']
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    logger.info(f"Created dataloader with {len(dataset)} clips")

    logger.info("Initializing model...")
    model = VideoAutoencoder(
        input_channels=3,
        latent_dim=256
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate']
    )

    logger.info("Starting training...")
    try:
        pretrain_autoencoder(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            num_epochs=CONFIG['num_epochs'],
            device=device
        )
        logger.info("Training completed successfully!")

        save_path = Path('models/autoencoder_test.pth')
        save_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    logger.info("Testing reconstruction...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(loader))
        if isinstance(test_batch, tuple):
            test_batch = test_batch[0]
        test_batch = test_batch.to(device)
        encoded, decoded = model(test_batch)

        reconstruction_error = torch.nn.functional.mse_loss(decoded, test_batch)
        logger.info(f"Test batch reconstruction error: {reconstruction_error.item():.6f}")


if __name__ == "__main__":
    main()
import datetime
from utils import setup_logging
from utils import get_video_paths
from utils import evaluate_model

from training.datasets import create_dataloaders
from training.train import train_and_evaluate
from models.autoencoder import VideoAutoencoder
import torch

logger = setup_logging(__name__)

def main():
    startdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Configuration - removed unnecessary params and kept only required ones
    CONFIG = {
        'base_path': './data/UBI_FIGHTS',
        'subset_size': 336,
        'batch_size': 48,
        'num_epochs': 9,
        'learning_rate': 4e-4,
        'input_channels': 3,
        'latent_dim': 256,
        'reconstruction_threshold': 0.015,
        'eval_interval': 4,
        'num_workers': 4,
        'pin_memory': True
    }

    # Load paths
    use_dvs = input("Use DVS-converted videos? (y/n): ").strip().lower() == 'y'
    train_paths = get_video_paths(CONFIG['base_path'], CONFIG['subset_size'], use_dvs, split="train")
    val_paths = get_video_paths(CONFIG['base_path'], CONFIG['subset_size'], use_dvs, split="val")
    test_paths = get_video_paths(CONFIG['base_path'], CONFIG['subset_size'], use_dvs, split="test")

    logger.info(f"Found {len(train_paths)} training videos, {len(val_paths)} validation videos, "
                f"and {len(test_paths)} test videos")

    # Create datasets and loaders - using new dataset implementation
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, val_paths, test_paths, CONFIG
    )

    # Initialize model
    model = VideoAutoencoder(
        input_channels=CONFIG['input_channels'],
        latent_dim=CONFIG['latent_dim']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    logger.info("Starting training...")
    logger.info(f"Model parameters: input_channels={CONFIG['input_channels']}, "
                f"latent_dim={CONFIG['latent_dim']}")
    logger.info(f"Training parameters: batch_size={CONFIG['batch_size']}, "
                f"learning_rate={CONFIG['learning_rate']}, "
                f"num_epochs={CONFIG['num_epochs']}")

    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=CONFIG['num_epochs'],
        device=device,
        eval_interval=CONFIG['eval_interval'],
        config=CONFIG,
        startdate=startdate,
        dvs=use_dvs,
        snapshot_dir="./snapshots"
    )

    logger.info("Starting final evaluation on the test set...")
    accuracy, mean_loss = evaluate_model(model, test_loader, device, CONFIG['reconstruction_threshold'])
    logger.info(f"Final Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Final Test Mean Loss: {mean_loss:.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Program interrupted by the user. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}")
        raise
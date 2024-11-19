import datetime
from utils import setup_logging
from utils import get_video_paths
from utils import evaluate_model

from training import create_dataloaders
from training import train_and_evaluate
from models.autoencoder import VideoAutoencoder
import torch

logger = setup_logging(__name__)

def main():
    startdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Fixed program start date
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Configuration
    CONFIG = {
        'base_path': './data/UBI_FIGHTS',
        'subset_size': 336,
        'batch_size': 48,
        'num_epochs': 9,
        'learning_rate': 4e-4,
        'clip_length': 16,
        'input_channels': 3,
        'latent_dim': 256,
        'target_size': (64, 64),
        'reconstruction_threshold': 0.015,
        'eval_interval': 4,
    }

    # Load paths
    use_dvs = input("Use DVS-converted videos? (y/n): ").strip().lower() == 'y'
    train_paths = get_video_paths(CONFIG['base_path'], CONFIG['subset_size'], use_dvs, split="train")
    val_paths = get_video_paths(CONFIG['base_path'], CONFIG['subset_size'], use_dvs, split="val")
    test_paths = get_video_paths(CONFIG['base_path'], CONFIG['subset_size'], use_dvs, split="test")

    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, val_paths, test_paths, CONFIG
    )

    # Initialize model
    model = VideoAutoencoder(
        input_channels=CONFIG['input_channels'],
        latent_dim=CONFIG['latent_dim']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    logger.info(f"Starting training with evaluation every {CONFIG['eval_interval']} epochs...")
    train_and_evaluate(
        model, train_loader, val_loader, optimizer,
        CONFIG['num_epochs'], device, CONFIG['eval_interval'],
        CONFIG, startdate, dvs=use_dvs
    )

    logger.info("Starting final evaluation on the test set...")
    accuracy, mean_loss, jaccard, cm = evaluate_model(model, test_loader, device, CONFIG['reconstruction_threshold'])
    logger.info(f"Final Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Final Test Mean Loss: {mean_loss:.6f}")
    logger.info(f"Final Test Jaccard Index: {jaccard:.6f}")
    logger.info(f"Final Test Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Program interrupted by the user. Exiting gracefully.")

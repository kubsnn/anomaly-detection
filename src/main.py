import argparse
import datetime
import os
import json
import torch
import random
from utils import setup_logging, evaluate_model, list_available_models, select_model
from training import create_dataloaders, train_and_evaluate
from testing import evaluate_with_metrics, log_metrics
from models.autoencoder import VideoAutoencoder

logger = setup_logging(__name__)


def prepare_dataset_paths(CONFIG):
    """
    Prepare dataset paths using the existing splits from preprocessing:
    - If use_dvs is True, use './data/UBI_FIGHTS/v2e/dataset/*'
    - Otherwise, use './data/UBI_FIGHTS/dataset/*'
    - Uses existing train/val/test splits
    """
    if CONFIG['use_dvs']:
        base_dir = os.path.join(CONFIG['base_path'], 'v2e', 'dataset')
    else:
        base_dir = os.path.join(CONFIG['base_path'], 'dataset')

    # Get paths for each split
    train_normal = [os.path.join(base_dir, 'train', 'normal', f)
                    for f in os.listdir(os.path.join(base_dir, 'train', 'normal'))
                    if f.endswith(('.mp4', '.avi'))]

    val_normal = [os.path.join(base_dir, 'val', 'normal', f)
                  for f in os.listdir(os.path.join(base_dir, 'val', 'normal'))
                  if f.endswith(('.mp4', '.avi'))]

    test_normal = [os.path.join(base_dir, 'test', 'normal', f)
                   for f in os.listdir(os.path.join(base_dir, 'test', 'normal'))
                   if f.endswith(('.mp4', '.avi'))]

    test_fight = [os.path.join(base_dir, 'test', 'fight', f)
                  for f in os.listdir(os.path.join(base_dir, 'test', 'fight'))
                  if f.endswith(('.mp4', '.avi'))]

    if not train_normal:
        logger.error("No training videos found.")
        raise FileNotFoundError("No training videos found in the dataset splits.")

    # Apply subset size limit if specified
    if CONFIG.get('subset_size'):
        train_normal = train_normal[:CONFIG['subset_size']]
        val_normal = val_normal[:CONFIG['subset_size'] // 5]  # Keeping roughly the same ratio
        test_videos_per_class = CONFIG['subset_size'] // 5
        test_normal = test_normal[:test_videos_per_class]
        test_fight = test_fight[:test_videos_per_class]

    CONFIG['train_paths'] = train_normal
    CONFIG['val_paths'] = val_normal
    CONFIG['test_paths'] = test_normal + test_fight

    logger.info("Dataset splits loaded:")
    logger.info(f"Train (normal): {len(train_normal)} videos")
    logger.info(f"Validation (normal): {len(val_normal)} videos")
    logger.info(f"Test: {len(test_normal)} normal, {len(test_fight)} fight videos")

def load_snapshot_model(CONFIG, device, snapshot_name):
    try:
        snapshot_path = os.path.join(CONFIG['snapshot_dir'], f"{snapshot_name}.pth")
        config_path = snapshot_path.replace(".pth", ".json")
        optimizer_path = snapshot_path.replace(".pth", "-optimizer.pth")

        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return None, None, None

        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        # Load model state
        model = VideoAutoencoder(
            input_channels=loaded_config['input_channels'],
            latent_dim=loaded_config['latent_dim']
        ).to(device)
        model.load_state_dict(torch.load(snapshot_path, map_location=device))

        # Load optimizer state
        optimizer = torch.optim.Adam(model.parameters(), lr=loaded_config['learning_rate'])
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            logger.info(f"Optimizer state loaded from: {optimizer_path}")
        else:
            logger.warning("Optimizer snapshot not found. Optimizer will start fresh.")

        return model, optimizer, loaded_config

    except FileNotFoundError as e:
        logger.error(str(e))
        return None, None, None


def initialize_new_model(CONFIG, device):
    logger.info("Starting a new training session.")
    model = VideoAutoencoder(
        input_channels=CONFIG['input_channels'],
        latent_dim=CONFIG['latent_dim']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    return model, optimizer, CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the Video Autoencoder.")
    parser.add_argument('--load', type=str, help="Load a specific snapshot by its name (without extension).")
    args = parser.parse_args()

    startdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Fixed program start date
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Create file to indicate the process started
    with open(f"started-{startdate}", "w") as f:
        f.write(f"{startdate}")

    # Configuration
    CONFIG = {
        'base_path': './data/UBI_FIGHTS',
        'subset_size': 10,
        'batch_size': 52,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'clip_length': 16,
        'input_channels': 1,
        'latent_dim': 256,
        'target_size': (64, 64),
        'reconstruction_threshold': 0.015,
        'eval_interval': 5,
        'snapshot_dir': './snapshots',
    }

    if args.load:
        model, optimizer, CONFIG = load_snapshot_model(CONFIG, device, args.load)
        if model is None or optimizer is None:
            logger.error("Failed to load the specified snapshot. Exiting.")
            return
    else:
        load_snapshot = input("Do you want to load a previous snapshot? (y/n): ").strip().lower() == 'y'
        if load_snapshot:
            available_models = list_available_models(CONFIG['snapshot_dir'])
            selected_model = select_model(available_models)
            model, optimizer, CONFIG = load_snapshot_model(CONFIG, device, selected_model)
            if model is None or optimizer is None:
                logger.error("Failed to load the selected snapshot. Exiting.")
                return
        else:
            use_dvs = input("Use DVS-converted videos? (y/n): ").strip().lower() == 'y'
            CONFIG['use_dvs'] = use_dvs
            prepare_dataset_paths(CONFIG)
            model, optimizer, CONFIG = initialize_new_model(CONFIG, device)

        # Create datasets and loaders using paths from CONFIG
        train_loader, val_loader, test_loader = create_dataloaders(
            CONFIG['train_paths'], CONFIG['val_paths'],  # Now using val_paths instead of test_paths
            CONFIG['test_paths'], CONFIG
        )

    if not optimizer:
        logger.fatal("Optimizer not initialized. Exiting.")
        return

    logger.info(f"Starting training with evaluation every {CONFIG['eval_interval']} epochs...")
    train_and_evaluate(
        model, train_loader, val_loader, optimizer,
        CONFIG['num_epochs'], device, CONFIG['eval_interval'],
        CONFIG, startdate, dvs=CONFIG['use_dvs']
    )

    logger.info("Starting final evaluation on the test set...")
    metrics = evaluate_with_metrics(model, test_loader, device)
    log_metrics(metrics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Program interrupted by the user. Exiting gracefully.")

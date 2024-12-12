# main.py
import datetime
import os
import json
import torch
import random
from utils import setup_logging
from utils import evaluate_model
from utils import list_available_models
from utils import select_model
from training import create_dataloaders, train_and_evaluate
from testing import evaluate_with_metrics, log_metrics
from models.autoencoder import VideoAutoencoder

logger = setup_logging(__name__)

def prepare_dataset_paths(CONFIG):
    """
    Prepare dataset paths:
    - If use_dvs is True, use './data/UBI_FIGHTS/v2e/videos/normal' and '.../fight'
      Otherwise, use './data/UBI_FIGHTS/videos/normal' and '.../fight'
    - Perform 80/20 split on normal videos for train/test.
    - Add all fight videos to the test set.
    """

    if CONFIG['use_dvs']:
        normal_dir = os.path.join(CONFIG['base_path'], 'v2e', 'videos', 'normal')
        fight_dir = os.path.join(CONFIG['base_path'], 'v2e', 'videos', 'fight')
    else:
        normal_dir = os.path.join(CONFIG['base_path'], 'videos', 'normal')
        fight_dir = os.path.join(CONFIG['base_path'], 'videos', 'fight')

    normal_videos = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.mp4', '.avi'))]
    fight_videos = [os.path.join(fight_dir, f) for f in os.listdir(fight_dir) if f.endswith(('.mp4', '.avi'))]

    if not normal_videos:
        logger.error("No normal videos found.")
        raise FileNotFoundError("No normal videos found.")

    random.shuffle(normal_videos)
    if CONFIG.get('subset_size') and CONFIG['subset_size'] < len(normal_videos):
        normal_videos = normal_videos[:CONFIG['subset_size']]

    # 80-20 split for normal videos
    total_normal = len(normal_videos)
    train_count = int(total_normal * 0.8)
    train_paths = normal_videos[:train_count]
    test_paths = normal_videos[train_count:total_normal]
    # Add fight videos entirely to the test set up to the size of the normal test set
    test_paths.extend(fight_videos[:len(test_paths)])

    CONFIG['train_paths'] = train_paths
    CONFIG['test_paths'] = test_paths

    logger.info(f"Dataset split: {len(train_paths)} normal train, {len(test_paths)} test (normal + fight).")


def load_or_initialize_model(CONFIG, device, startdate):
    load_snapshot = input("Do you want to load a previous snapshot? (y/n): ").strip().lower() == 'y'
    if load_snapshot:
        model, optimizer, loaded_config = load_snapshot_model(CONFIG, device)
        if model is not None and optimizer is not None and loaded_config is not None:
            CONFIG.update(loaded_config)
            logger.info("Loaded configuration from snapshot.")
            return model, optimizer, CONFIG
        else:
            return None, None, CONFIG
    else:
        # Prompt about use_dvs only if we are starting fresh
        use_dvs = True #input("Use DVS-converted videos? (y/n): ").strip().lower() == 'y'
        CONFIG['use_dvs'] = use_dvs

        # Prepare dataset paths based on new setting
        prepare_dataset_paths(CONFIG)
        return initialize_new_model(CONFIG, device)


def load_snapshot_model(CONFIG, device):
    try:
        available_models = list_available_models(CONFIG['snapshot_dir'])
        selected_model = select_model(available_models)
        logger.info(f"Loading model from snapshot: {selected_model}")

        # Load associated config
        config_path = selected_model.replace(".pth", ".json")
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
        model.load_state_dict(torch.load(selected_model, map_location=device))

        # Load optimizer state
        optimizer = torch.optim.Adam(model.parameters(), lr=loaded_config['learning_rate'])
        optimizer_path = selected_model.replace(".pth", "-optimizer.pth")
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
    startdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Fixed program start date
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Configuration
    CONFIG = {
        'base_path': './data/UBI_FIGHTS',
        'subset_size': 100,
        'batch_size': 32,
        'num_epochs': 7,
        'learning_rate': 2e-4,
        'clip_length': 16,
        'input_channels': 3,
        'latent_dim': 256,
        'target_size': (64, 64),
        'reconstruction_threshold': 0.015,
        'eval_interval': 3,
        'snapshot_dir': './snapshots',
    }

    model, optimizer, CONFIG = load_or_initialize_model(CONFIG, device, startdate)
    if model is None or optimizer is None:
        logger.error("Model or optimizer not initialized. Exiting.")
        return

    # Create datasets and loaders using paths from CONFIG
    # No val_paths in this scenario, just train and test
    train_loader, val_loader, test_loader = create_dataloaders(
        CONFIG['train_paths'], CONFIG['test_paths'],
        CONFIG['test_paths'], CONFIG
    )

    # Train and evaluate if newly initialized or continuing training

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

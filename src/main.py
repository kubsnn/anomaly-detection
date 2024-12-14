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
    Prepare dataset paths:
    - If use_dvs is True, use './data/UBI_FIGHTS/v2e/videos/normal' and '.../fight'
      Otherwise, use './data/UBI_FIGHTS/videos/normal' and '.../fight'
    - Perform 80/20 split on normal videos for train/test.
    - Ensure the test set is balanced between normal and fight videos.
    - Add excess normal videos (beyond the fight video count) to the training set.
    - Log detailed information about the split percentages.
    """
    if CONFIG['use_dvs']:
        normal_dir = os.path.join(CONFIG['base_path'], 'v2e', 'videos', 'normal', 'split')
        fight_dir = os.path.join(CONFIG['base_path'], 'v2e', 'videos', 'fight', 'split')
    else:
        normal_dir = os.path.join(CONFIG['base_path'], 'videos', 'normal', 'split')
        fight_dir = os.path.join(CONFIG['base_path'], 'videos', 'fight', 'split')

    normal_videos = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.mp4')]
    fight_videos = [os.path.join(fight_dir, f) for f in os.listdir(fight_dir) if f.endswith('.mp4')]
    total_fight = len(fight_videos)
    if not normal_videos:
        logger.error("No normal videos found.")
        raise FileNotFoundError("No normal videos found.")

    random.shuffle(normal_videos)
    random.shuffle(fight_videos)

    if CONFIG.get('subset_size') and CONFIG['subset_size'] < len(normal_videos):
        normal_videos = normal_videos[:CONFIG['subset_size']]

    # 80-20 split for normal videos
    total_normal = len(normal_videos)
    train_count = int(total_normal * 0.8)
    train_paths = normal_videos[:train_count]
    normal_test_paths = normal_videos[train_count:total_normal]

    # Balance the test set: limit normal and fight videos to the same size
    num_fight_test = len(fight_videos)
    num_normal_test = len(normal_test_paths)

    if num_fight_test > num_normal_test:
        fight_videos = fight_videos[:num_normal_test]
    else:
        normal_test_paths = normal_test_paths[:num_fight_test]

    # Add excess normal test videos to the train set
    excess_normal_test_paths = normal_test_paths[num_fight_test:]
    train_paths.extend(excess_normal_test_paths)

    # Combine balanced normal test paths with fight videos for the test set
    test_paths = normal_test_paths[:num_fight_test] + fight_videos

    # Log detailed split percentages
    total_videos = len(normal_videos) + len(fight_videos)
    train_percentage = (len(train_paths) / total_videos) * 100
    test_percentage = (len(test_paths) / total_videos) * 100
    normal_train_percentage = (len(train_paths) - len(excess_normal_test_paths)) / len(normal_videos) * 100
    normal_test_percentage = len(normal_test_paths[:num_fight_test]) / len(normal_videos) * 100
    fight_test_percentage = len(fight_videos) / total_fight * 100

    logger.info(f"Dataset split:")
    logger.info(f"- Total videos: {total_videos}")
    logger.info(f"- Train set: {len(train_paths)} ({train_percentage:.2f}%)")
    logger.info(f"  - Normal videos in train set: {len(train_paths) - len(excess_normal_test_paths)} ({normal_train_percentage:.2f}%)")
    logger.info(f"- Test set: {len(test_paths)} ({test_percentage:.2f}%)")
    logger.info(f"  - Normal videos in test set: {len(normal_test_paths[:num_fight_test])} ({normal_test_percentage:.2f}%)")
    logger.info(f"  - Fight videos in test set: {len(fight_videos)} ({fight_test_percentage:.2f}%)")

    CONFIG['train_paths'] = train_paths
    CONFIG['test_paths'] = test_paths



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
    parser.add_argument('--no-load', action='store_true', help="Do not load any snapshot.")
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
        'subset_size': 12500,
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 5e-3,
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
        if args.no_load:
            load_snapshot = False
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
            use_dvs = True #input("Use DVS-converted videos? (y/n): ").strip().lower() == 'y'
            CONFIG['use_dvs'] = use_dvs
            prepare_dataset_paths(CONFIG)
            model, optimizer, CONFIG = initialize_new_model(CONFIG, device)

    # Create datasets and loaders using paths from CONFIG
    train_loader, val_loader, test_loader = create_dataloaders(
        CONFIG['train_paths'], CONFIG['test_paths'],
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

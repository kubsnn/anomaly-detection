import glob
import os
import json
import torch
from .logger import setup_logging

logger = setup_logging(__name__)


def save_snapshot(model, optimizer, config, snapshot_dir, startdate, best=False, dvs=False) -> None:
    os.makedirs(snapshot_dir, exist_ok=True)
    prefix = f"{startdate}-best" if best else f"{startdate}-final"
    if dvs:
        prefix = f"dvs-{prefix}"
        
    model_path = os.path.join(snapshot_dir, f"{prefix}.pth")
    config_path = os.path.join(snapshot_dir, f"{prefix}.json")
    optimizer_path = os.path.join(snapshot_dir, f"{prefix}-optimizer.pth")

    # Save model state
    torch.save(model.state_dict(), model_path)

    # Save optimizer state
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    logger.info(f"Snapshot saved: {model_path}, {optimizer_path}")


def list_available_models(snapshot_dir: str) -> list:
    """
    List all available model snapshots, sorted by date.

    Args:
        snapshot_dir (str): Path to the directory containing snapshots.

    Returns:
        list: Sorted list of available models.
    """
    model_paths = glob.glob(f"{snapshot_dir}/*.pth")
    model_paths = [path for path in model_paths if not path.endswith("optimizer.pth")]
    model_paths = sorted(model_paths, key=os.path.getmtime, reverse=True)
    if not model_paths:
        logger.error("No models found in the snapshot directory.")
        raise FileNotFoundError("No models found in the snapshot directory.")
    return model_paths


def select_model(models: list) -> str:
    """
    Display available models and let the user select one.

    Args:
        models (list): List of available models.

    Returns:
        str: Selected model path.
    """
    logger.info("\nAvailable Models:")
    for idx, model in enumerate(models, start=1):
        logger.info(f"{idx}: {os.path.basename(model)}")

    while True:
        try:
            choice = int(input("\nEnter the number of the model to evaluate: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            else:
                logger.warning("Invalid selection. Please choose a valid model number.")
        except ValueError:
            logger.warning("Invalid input. Please enter a number.")
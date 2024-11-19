import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from data.clipper import VideoClipDataset
from models.autoencoder import VideoAutoencoder
from utils import setup_logging
from utils import get_video_paths
from utils import evaluate_model
import json
import glob
from tqdm import tqdm

logger = setup_logging(__name__)

def jaccard_index(y_true, y_pred):
    intersection = torch.sum((y_true == 1) & (y_pred == 1)).item()
    union = torch.sum((y_true == 1) | (y_pred == 1)).item()
    return intersection / union if union != 0 else 0

def confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def load_model(model_path: str, config_path: str, device: torch.device):
    """
    Load the model and its configuration.

    Args:
        model_path (str): Path to the saved model (.pth).
        config_path (str): Path to the configuration file (.json).
        device (torch.device): Device to load the model onto.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
        config (dict): Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = VideoAutoencoder(
        input_channels=config['input_channels'],
        latent_dim=config['latent_dim']
    ).to(device)

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    config['base_path'] = os.path.join("../..", config['base_path'])
    return model, config


def prepare_validation_loader(config: dict, use_dvs: bool):
    """
    Prepare the validation DataLoader.

    Args:
        config (dict): Configuration dictionary.
        use_dvs (bool): Whether to use DVS-converted videos.

    Returns:
        DataLoader: Validation DataLoader.
    """
    val_video_paths = get_video_paths(
        base_path=config['base_path'],
        subset_size=config['subset_size'],
        use_dvs=use_dvs,
        split="val"
    )

    if not val_video_paths:
        logger.error("No validation videos found. Check the dataset structure.")
        raise ValueError("No validation videos found. Check the dataset structure.")

    val_dataset = VideoClipDataset(
        video_paths=val_video_paths,
        clip_length=config['clip_length'],
        clip_overlap=0.5,
        min_clips=1,
        augment=False,
        target_size=config['target_size']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Prepared validation dataset with {len(val_dataset)} clips.")
    return val_loader


def evaluate_with_metrics(model, loader, device, threshold=0.01, num_classes=2):
    """
    Evaluate the model and calculate accuracy and mean loss.

    Args:
        model (torch.nn.Module): Trained autoencoder model.
        loader (DataLoader): Validation DataLoader.
        device (torch.device): Device for evaluation.
        threshold (float): Reconstruction threshold for anomaly detection.

    Returns:
        dict: Dictionary containing accuracy and mean loss.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if isinstance(batch, tuple):  # Handle datasets with labels
                batch, labels = batch
            else:
                labels = torch.zeros(batch.size(0))  # Default to 0 (non-anomaly)
            
            _, reconstructed = model(batch)
            mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2, 3, 4))
            total_loss += mse.sum().item()
            preds = (mse < threshold).long()
            correct += preds.sum().item()
            total += batch.size(0)

            all_y_true.append(labels)
            all_y_pred.append(preds)
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    mean_loss = total_loss / total if total > 0 else 0
    
    y_true = torch.cat(all_y_true)
    y_pred = torch.cat(all_y_pred)
    jaccard = jaccard_index(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    return {
        "accuracy": accuracy,
        "mean_loss": mean_loss,
        "jaccard": jaccard,
        "confusion_matrix": cm
    }


def list_available_models(snapshot_dir: str):
    """
    List all available model snapshots, sorted by date.

    Args:
        snapshot_dir (str): Path to the directory containing snapshots.

    Returns:
        list: Sorted list of available models.
    """
    model_paths = glob.glob(f"{snapshot_dir}/*.pth")
    model_paths = sorted(model_paths, key=os.path.getmtime, reverse=True)
    if not model_paths:
        logger.error("No models found in the snapshot directory.")
        raise FileNotFoundError("No models found in the snapshot directory.")
    return model_paths


def select_model(models: list):
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


def run_validation_evaluation(snapshot_dir: str, device: torch.device):
    """
    Run evaluation on the validation set using a selected model.

    Args:
        snapshot_dir (str): Directory containing model snapshots.
        device (torch.device): Device for evaluation.
    """
    # List and select model
    models = list_available_models(snapshot_dir)
    model_path = select_model(models)
    config_path = model_path.replace(".pth", ".json")

    # Determine if the model was trained on DVS data
    use_dvs = os.path.basename(model_path).startswith("dvs-")

    # Load model and configuration
    model, config = load_model(model_path, config_path, device)

    # Prepare the validation DataLoader
    val_loader = prepare_validation_loader(config, use_dvs)

    # Run evaluation
    logger.info(f"Starting validation evaluation on {'DVS' if use_dvs else 'default'} videos...")
    metrics = evaluate_with_metrics(
        model=model,
        loader=val_loader,
        device=device,
        threshold=config['reconstruction_threshold']
    )

    logger.info(f"Validation Metrics:")
    logger.info(f" - Accuracy (MSE < {config['reconstruction_threshold']}): {metrics['accuracy']:.2f}%")
    logger.info(f" - Mean Reconstruction Loss (MSE): {metrics['mean_loss']:.6f}")


if __name__ == "__main__":
    try:
        # Define the directory containing snapshots
        snapshot_dir = "../../snapshots"

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Initializing validation evaluation pipeline...")
        run_validation_evaluation(snapshot_dir, device)

    except KeyboardInterrupt:
        logger.warning("Program interrupted by the user. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Validation evaluation pipeline failed: {str(e)}")
        raise

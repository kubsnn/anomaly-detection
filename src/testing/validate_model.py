# validate_model.py

import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from data.clipper import VideoClipDataset
from models.autoencoder import VideoAutoencoder
from utils import setup_logging
from utils import get_video_paths
from tqdm import tqdm
import json
import glob

logger = setup_logging(__name__)

def jaccard_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate the Jaccard Index between true and predicted masks."""
    intersection = torch.sum((y_true == 1) & (y_pred == 1)).item()
    union = torch.sum((y_true == 1) | (y_pred == 1)).item()
    return intersection / union if union != 0 else 0

def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute the confusion matrix."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def compute_binary_classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Compute accuracy, precision, recall, and F1 score for binary classification."""
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy * 100,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def find_best_threshold(mse_scores: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Find the best threshold based on F1 Score.

    Args:
        mse_scores (torch.Tensor): Reconstruction errors for all samples.
        labels (torch.Tensor): True labels for all samples.

    Returns:
        tuple: Best threshold and dictionary of metrics.
    """
    thresholds = torch.linspace(mse_scores.min(), mse_scores.max(), steps=100)
    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_metrics = None

    for threshold in thresholds:
        preds = (mse_scores > threshold).long()
        metrics = compute_binary_classification_metrics(labels, preds)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold.item()
            best_metrics = metrics

    return best_threshold, best_metrics

def load_model(model_path: str, config_path: str, device: torch.device):
    """Load the model and its configuration."""
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

def prepare_validation_loader(config: dict, use_dvs: bool) -> DataLoader:
    """Prepare the validation DataLoader including both 'normal' and 'fight' videos."""
    # Get video paths from both 'normal' and 'fight' directories
    normal_val_paths = get_video_paths(
        base_path=config['base_path'],
        subset_size=config['subset_size'],
        use_dvs=use_dvs,
        split="val",
        type="normal"
    )

    fight_val_paths = get_video_paths(
        base_path=config['base_path'],
        subset_size=config['subset_size'],
        use_dvs=use_dvs,
        split="val",
        type="fight"
    )

    # Combine both lists of video paths
    val_video_paths = normal_val_paths + fight_val_paths

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

def evaluate_with_metrics(model, loader, device, num_classes=2):
    model.eval()
    total_loss = 0
    all_mse = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            frames, labels = batch  # Unpack frames and labels
            frames = frames.to(device)
            labels = labels.to(device)

            # Forward pass
            _, reconstructed = model(frames)
            mse = torch.mean((frames - reconstructed) ** 2, dim=(1, 2, 3, 4))
            total_loss += mse.sum().item()

            all_mse.append(mse.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all MSEs and labels
    all_mse = torch.cat(all_mse)
    all_labels = torch.cat(all_labels)

    mean_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0

    # Determine the best threshold
    best_threshold, best_metrics = find_best_threshold(all_mse, all_labels)

    # Compute confusion matrix with the best threshold
    y_pred = (all_mse > best_threshold).long()
    y_true = all_labels.long()

    cm = confusion_matrix(y_true, y_pred, num_classes)
    jaccard = jaccard_index(y_true, y_pred)

    return {
        "mean_loss": mean_loss,
        "jaccard": jaccard,
        "confusion_matrix": cm,
        "best_threshold": best_threshold,
        **best_metrics  # Include accuracy, precision, recall, f1_score
    }

def list_available_models(snapshot_dir: str):
    """List all available model snapshots, sorted by date."""
    model_paths = glob.glob(f"{snapshot_dir}/*.pth")
    model_paths = sorted(model_paths, key=os.path.getmtime, reverse=True)
    if not model_paths:
        logger.error("No models found in the snapshot directory.")
        raise FileNotFoundError("No models found in the snapshot directory.")
    return model_paths

def select_model(models: list) -> str:
    """Display available models and let the user select one."""
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
    """Run evaluation on the validation set using a selected model."""
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
        device=device
    )

    logger.info(f"Validation Metrics:")
    logger.info(f" - Best Threshold: {metrics['best_threshold']:.6f}")
    logger.info(f" - Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f" - Precision: {metrics['precision']:.4f}")
    logger.info(f" - Recall: {metrics['recall']:.4f}")
    logger.info(f" - F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f" - Mean Reconstruction Loss (MSE): {metrics['mean_loss']:.6f}")
    logger.info(f" - Jaccard Index: {metrics['jaccard']:.4f}")
    logger.info(f" - Confusion Matrix:\n{metrics['confusion_matrix']}")

if __name__ == "__main__":
    try:
        # Define the directory containing snapshots
        snapshot_dir = "../../snapshots"

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info("Initializing validation evaluation pipeline...")
        run_validation_evaluation(snapshot_dir, device)

    except KeyboardInterrupt:
        logger.warning("Program interrupted by the user. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Validation evaluation pipeline failed: {str(e)}")
        raise

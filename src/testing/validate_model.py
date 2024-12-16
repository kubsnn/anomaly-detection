# validate_model.py

import os
import random
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from data.clipper import VideoClipDataset
from models.autoencoder import VideoAutoencoder
from utils import setup_logging
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

def get_best_threshold(mse_scores: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Find the best threshold based on F1 Score."""
    threshold = 0.02
    preds = (mse_scores > threshold).long()
    metrics = compute_binary_classification_metrics(labels, preds)
    best_f1 = metrics['f1_score']
    best_threshold = threshold
    best_metrics = metrics

    return best_threshold, best_metrics

def find_best_threshold(mse_scores: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Find the best threshold based on F1 Score."""
    thresholds = torch.linspace(mse_scores.min(), mse_scores.max(), steps=200)
    print(mse_scores.min(), mse_scores.max())
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

def find_balanced_threshold(mse_scores: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5) -> tuple:
    """
    Find the best threshold based on a balanced metric that combines F1 score 
    and the balance between precision and recall.

    Args:
        mse_scores (torch.Tensor): MSE scores from the autoencoder.
        labels (torch.Tensor): Ground truth labels (binary).
        alpha (float): Weight for balancing F1 score and precision-recall difference (0 to 1).
                      0.5 gives equal importance to F1 score and balance.

    Returns:
        tuple: (best_threshold, best_metrics) where metrics include accuracy, precision, recall, F1 score, etc.
    """
    thresholds = torch.linspace(mse_scores.min(), mse_scores.max(), steps=200)
    print(mse_scores.min(), mse_scores.max())
    best_score = 0.0
    best_threshold = thresholds[0]
    best_metrics = None

    for threshold in thresholds:
        preds = (mse_scores > threshold).long()
        metrics = compute_binary_classification_metrics(labels, preds)
        
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1_score']
        
        # Add a penalty for imbalanced precision and recall
        precision_recall_balance = 1 - abs(precision - recall)  # Maximum balance = 1 when precision == recall

        # Combine F1 score and balance metric
        combined_score = alpha * f1_score + (1 - alpha) * precision_recall_balance

        if combined_score > best_score:
            best_score = combined_score
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
    # Adjust base_path if needed
    config['base_path'] = os.path.join("../..", config['base_path'])
    print(config['base_path'])

    # # add ../.. to every path in the config
    # for i in range(len(config['test_paths'])):
    #     # Extract the directory and filename from the existing path
    #     dir_path, filename = os.path.split(config['test_paths'][i])
    #     # Construct the new path with 'split' directory
    #     new_path = os.path.join(dir_path, 'split', filename)
    #     # Update the path in the config
    #     config['test_paths'][i] = os.path.join("../..", new_path)

    return model, config

def remove_90_percent_of_paths(paths):
    """
    Remove 90% of paths that start with 'N'.

    Args:
        paths (list): List of file paths.

    Returns:
        list: Filtered list of file paths.
    """
    n_paths = [path for path in paths if ('N' in path)]
    non_n_paths = [path for path in paths if not ('N' in path)]

    # Calculate the number of paths to remove
    num_to_remove = int(len(n_paths) * 0.98)

    # Randomly select paths to remove
    paths_to_remove = random.sample(n_paths, num_to_remove)

    # Filter out the selected paths
    filtered_paths = [path for path in paths if path not in paths_to_remove]

    return filtered_paths

def prepare_test_loader(config: dict) -> DataLoader:
    """
    Prepare the test DataLoader using paths from config['test_paths'].
    Previously, we tried to use validation paths, but now we rely on pre-defined splits.
    """
    if 'test_paths' not in config or not config['test_paths']:
        logger.error("No test paths found in config. Check the dataset preparation steps.")
        raise ValueError("No test paths found in config.")

    test_paths = config['test_paths']
    
    # remove 98% of paths that starts with 'N'
   # test_paths = remove_90_percent_of_paths(test_paths)
    # add '../..' to the paths
    test_paths = [os.path.join("../..", path) for path in test_paths]


    # Create the dataset from these test paths
    test_dataset = VideoClipDataset(
        video_paths=test_paths,
        clip_length=config['clip_length'],
        clip_overlap=0.5,
        min_clips=1,
        augment=False,
        target_size=config['target_size']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    logger.info(f"Prepared test dataset with {len(test_dataset)} clips.")
    return test_loader

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
    best_threshold, best_metrics = find_balanced_threshold(all_mse, all_labels)

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
    """Run evaluation on the test set using a selected model."""
    # List and select model
    models = list_available_models(snapshot_dir)
    model_path = select_model(models)
    config_path = model_path.replace(".pth", ".json")

    # Load model and configuration
    model, config = load_model(model_path, config_path, device)

    # Prepare the test DataLoader
    test_loader = prepare_test_loader(config)

    # Run evaluation
    logger.info(f"Starting evaluation on {'DVS' if config.get('use_dvs', False) else 'default'} videos...")
    metrics = evaluate_with_metrics(
        model=model,
        loader=test_loader,
        device=device
    )

    # Log and display the evaluation metrics
    log_metrics(metrics)

def log_metrics(metrics):
    logger.info(f"Evaluation Metrics:")
    logger.info(f">  Best Threshold: {metrics['best_threshold']:.6f}")
    logger.info(f">  Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f">  Precision: {metrics['precision']:.4f}")
    logger.info(f">  Recall: {metrics['recall']:.4f}")
    logger.info(f">  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f">  Mean Reconstruction Loss (MSE): {metrics['mean_loss']:.6f}")
    logger.info(f">  Jaccard Index: {metrics['jaccard']:.4f}")
    logger.info(f">  Confusion Matrix:\n{metrics['confusion_matrix']}")

if __name__ == "__main__":
    try:
        # Define the directory containing snapshots
        snapshot_dir = "../../snapshots"

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info("Initializing evaluation pipeline...")
        run_validation_evaluation(snapshot_dir, device)

    except KeyboardInterrupt:
        logger.warning("Program interrupted by the user. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        raise

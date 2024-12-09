import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
from pathlib import Path

from datasets import ProcessedVideoDataset
from models.autoencoder import FightDetectionAutoencoder
from utils.training import evaluate_model, compute_reconstruction_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_autoencoder(model_path, test_video_paths, device, config):
    """
    Evaluate a trained autoencoder model.

    Args:
        model_path (str): Path to the saved model checkpoint
        test_video_paths (list): List of paths to test videos
        device (torch.device): Device to run evaluation on
        config (dict): Configuration dictionary containing model parameters
    """
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model
    model = FightDetectionAutoencoder(
        input_channels=config['input_channels'],
        latent_dim=config['latent_dim']
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test dataset - using new ProcessedVideoDataset
    test_dataset = ProcessedVideoDataset(test_video_paths)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Compute reconstruction errors
    logger.info("Computing reconstruction errors...")
    reconstruction_errors = compute_reconstruction_error(model, test_loader, device)

    # Calculate threshold (using mean + 2*std as example threshold)
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
    logger.info(f"Calculated threshold: {threshold:.6f}")

    # Evaluate model
    logger.info("Evaluating model...")
    evaluation_results = evaluate_model(model, test_loader, threshold, device)

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info(f"Mean Reconstruction Error: {np.mean(reconstruction_errors):.6f}")
    logger.info(f"Max Reconstruction Error: {np.max(reconstruction_errors):.6f}")
    logger.info(f"Min Reconstruction Error: {np.min(reconstruction_errors):.6f}")

    if 'confusion_matrix' in evaluation_results:
        cm = evaluation_results['confusion_matrix']
        logger.info("\nConfusion Matrix:")
        logger.info(f"True Positives: {cm['tp']}")
        logger.info(f"False Positives: {cm['fp']}")
        logger.info(f"True Negatives: {cm['tn']}")
        logger.info(f"False Negatives: {cm['fn']}")

        logger.info("\nMetrics:")
        logger.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"F1 Score: {evaluation_results['f1_score']:.4f}")

    return evaluation_results, reconstruction_errors


if __name__ == "__main__":
    # Get the appropriate device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load your configuration
    model_path = 'models/fight_detection_autoencoder.pth'
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Get test video paths
    test_video_paths = [
        "/Users/morpheus/Documents/PolitechnikaPoznanska/Semestr7/WykrywanieAnomaliiWZachowaniachSpolecznych/UBI_FIGHTS/videos/normal/N_0_0_0_1_0.mp4"
    ]

    # Run evaluation
    try:
        results, errors = evaluate_autoencoder(model_path, test_video_paths, device, config)

        # Save results
        save_dir = Path('evaluation_results')
        save_dir.mkdir(exist_ok=True)

        np.save(save_dir / 'reconstruction_errors.npy', errors)
        np.save(save_dir / 'evaluation_metrics.npy', results)

        logger.info(f"Results saved to {save_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise
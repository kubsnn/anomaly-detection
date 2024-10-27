import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pretrain_autoencoder(model, train_loader, val_loader=None, num_epochs=10,
                         learning_rate=1e-4, device='cuda', optimizer=None):
    """
    Pretrain the video autoencoder.

    Args:
        model (nn.Module): The video autoencoder model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for the optimizer (used only if optimizer is None)
        device (str): Device to train on ('cuda', 'mps', or 'cpu')
        optimizer (torch.optim.Optimizer, optional): Custom optimizer. If None, Adam will be used

    Returns:
        dict: Training history containing loss values
    """
    model = model.to(device)
    criterion = nn.MSELoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [] if val_loader else None
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            if isinstance(batch, tuple):
                data = batch[0]
            else:
                data = batch

            data = data.to(device)

            optimizer.zero_grad()
            encoded, decoded = model(data)
            loss = criterion(decoded, data)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'train_loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, tuple):
                        data = batch[0]
                    else:
                        data = batch

                    data = data.to(device)
                    encoded, decoded = model(data)
                    loss = criterion(decoded, data)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}"
            )

    return history


def compute_reconstruction_error(model, dataloader, device):
    """
    Compute reconstruction error for each video clip in the dataloader.

    Args:
        model (nn.Module): The trained autoencoder model
        dataloader (DataLoader): DataLoader containing the video clips
        device (str): Device to compute on ('cuda', 'mps', or 'cpu')

    Returns:
        np.array: Array of reconstruction errors for each clip
    """
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reconstruction errors"):
            if isinstance(batch, tuple):
                data = batch[0]
            else:
                data = batch

            data = data.to(device)
            encoded, decoded = model(data)

            error = nn.MSELoss(reduction='none')(decoded, data)

            error = error.mean(dim=(1, 2, 3, 4)).cpu().numpy()
            reconstruction_errors.extend(error)

    return np.array(reconstruction_errors)


def evaluate_model(model, test_loader, threshold, device):
    """
    Evaluate the model's anomaly detection performance.

    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader for test data
        threshold (float): Anomaly threshold
        device (str): Device to compute on ('cuda', 'mps', or 'cpu')

    Returns:
        dict: Dictionary containing evaluation metrics and frame-level predictions
    """
    model.eval()
    test_errors = compute_reconstruction_error(model, test_loader, device)
    predictions = (test_errors > threshold).astype(int)

    true_labels = []
    for batch in test_loader:
        if isinstance(batch, tuple) and len(batch) > 1:
            labels = batch[1]
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            true_labels.extend(labels)

    if not true_labels:
        logger.warning("No labels found in test_loader. Returning only predictions.")
        return {
            'predictions': predictions,
            'reconstruction_errors': test_errors
        }

    true_labels = np.array(true_labels)

    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()

    accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'predictions': predictions,
        'reconstruction_errors': test_errors,
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    }
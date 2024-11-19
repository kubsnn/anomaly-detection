from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from .logger import setup_logging

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

def evaluate_model(model, loader: DataLoader, device: torch.device, threshold: float, num_classes: int):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            _, reconstructed = model(batch)
            mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2, 3, 4))
            total_loss += mse.sum().item()
            preds = mse < threshold
            correct += preds.sum().item()
            total += batch.size(0)
            all_y_true.append(torch.ones_like(preds))
            all_y_pred.append(preds)
    accuracy = (correct / total) * 100 if total > 0 else 0
    mean_loss = total_loss / total if total > 0 else 0

    y_true = torch.cat(all_y_true)
    y_pred = torch.cat(all_y_pred)
    jaccard = jaccard_index(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    return accuracy, mean_loss, jaccard, cm

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from .logger import setup_logging

logger = setup_logging(__name__)


def evaluate_model(model, loader: DataLoader, device: torch.device, threshold: float):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            _, reconstructed = model(batch)
            mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2, 3, 4))
            total_loss += mse.sum().item()
            correct += (mse < threshold).sum().item()
            total += batch.size(0)

    accuracy = (correct / total) * 100 if total > 0 else 0
    mean_loss = total_loss / total if total > 0 else 0
    return accuracy, mean_loss

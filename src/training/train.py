from utils import setup_logging, save_snapshot, evaluate_model
from tqdm import tqdm
import torch
from testing import evaluate_with_metrics, log_metrics

logger = setup_logging(__name__)


def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs, device, eval_interval, config, startdate, dvs=False, snapshot_dir = "./snapshots"):
    best_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        model.train()
        for batch, _ in tqdm(train_loader, desc="Training"):
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = torch.mean((batch - reconstructed) ** 2)
            loss.backward()
            optimizer.step()

        if epoch % eval_interval == 0:
            logger.info(f"Validation {epoch // eval_interval}/{num_epochs // eval_interval}")
            
            best_f1 = evaluate_and_snapshot(model, val_loader, device, optimizer, config, startdate, dvs, snapshot_dir, best_f1)


def evaluate_and_snapshot(model, val_loader, device, optimizer, config, startdate, dvs, snapshot_dir, best_f1):
    metrics = evaluate_with_metrics(model, val_loader, device)
    log_metrics(metrics)

    save_snapshot(model, optimizer, config, snapshot_dir, startdate, best=False, dvs=dvs)

    if metrics["f1_score"] > best_f1:
        save_snapshot(model, optimizer, config, snapshot_dir, startdate, best=True, dvs=dvs)
        best_f1 = metrics["f1_score"]
    return metrics["f1_score"]

from utils import setup_logging, save_snapshot, evaluate_model
from tqdm import tqdm
import torch

logger = setup_logging(__name__)


def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs, device, eval_interval, config, startdate, dvs=False, snapshot_dir = "./snapshots"):
    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = torch.mean((batch - reconstructed) ** 2)
            loss.backward()
            optimizer.step()

        if epoch % eval_interval == 0:
            logger.info(f"Validation {epoch // eval_interval}/{num_epochs // eval_interval}")
            accuracy, mean_loss = evaluate_model(model, val_loader, device, config['reconstruction_threshold'])
            logger.info(f"Accuracy = {accuracy:.2f}%, Loss = {mean_loss:.6f}")
            save_snapshot(model, config, snapshot_dir, startdate, best=False, dvs=dvs)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_snapshot(model, config, snapshot_dir, startdate, best=True, dvs=dvs)

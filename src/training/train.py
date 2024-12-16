from utils import setup_logging, save_snapshot, evaluate_model
from tqdm import tqdm
import torch
from testing import evaluate_with_metrics, log_metrics

logger = setup_logging(__name__)

def update_weight_decay(optimizer, new_weight_decay):
    """
    Update weight decay for all parameter groups in the optimizer.
    """
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = new_weight_decay
    logger.info(f"Updated weight decay to: {new_weight_decay}")

from torch.optim.lr_scheduler import StepLR

def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs, device, eval_interval, config, startdate, dvs=False, snapshot_dir="./snapshots"):
    best_f1 = 0.0
    logger.info(f"Validation 0/{num_epochs // eval_interval}")
            
    best_f1 = evaluate_and_snapshot(model, val_loader, device, optimizer, config, startdate, dvs, snapshot_dir, best_f1)
    curr_epoch = config["current_epoch"]
    max_epochs = curr_epoch + num_epochs

    # Initialize the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=3, gamma=0.75)  # Halve LR every 5 epochs

    for epoch in range(1, num_epochs + 1):
        config["current_epoch"] = epoch + curr_epoch
        logger.info(f"Epoch {epoch}/{num_epochs}")
        model.train()
        
        epoch_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Training (Epoch {config['current_epoch']}/{max_epochs})", leave=False)
        
        for batch, _ in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = torch.mean((batch - reconstructed) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")

        # Step the scheduler to adjust the learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} - Learning Rate: {current_lr:.6e}")

        # Log average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Average Training Loss (Epoch {epoch}): {avg_epoch_loss:.6f}")

        if epoch % eval_interval == 0:
            logger.info(f"Validation {epoch // eval_interval}/{num_epochs // eval_interval}")
            best_f1 = evaluate_and_snapshot(model, val_loader, device, optimizer, config, startdate, dvs, snapshot_dir, best_f1)




def evaluate_and_snapshot(model, val_loader, device, optimizer, config, startdate, dvs, snapshot_dir, best_f1):
    save_snapshot(model, optimizer, config, snapshot_dir, startdate, best=False, dvs=dvs)

    metrics = evaluate_with_metrics(model, val_loader, device)
    log_metrics(metrics)


    if metrics["f1_score"] > best_f1:
        save_snapshot(model, optimizer, config, snapshot_dir, startdate, best=True, dvs=dvs)
        best_f1 = metrics["f1_score"]
    return metrics["f1_score"]

import os
import json
import torch
from .logger import setup_logging

logger = setup_logging(__name__)


def save_snapshot(model, optimizer, config, snapshot_dir, startdate, best=False, dvs=False):
    os.makedirs(snapshot_dir, exist_ok=True)
    prefix = f"{startdate}-best" if best else f"{startdate}-final"
    if dvs:
        prefix = f"dvs-{prefix}"
        
    model_path = os.path.join(snapshot_dir, f"{prefix}.pth")
    config_path = os.path.join(snapshot_dir, f"{prefix}.json")
    optimizer_path = os.path.join(snapshot_dir, f"{prefix}-optimizer.pth")

    # Save model state
    torch.save(model.state_dict(), model_path)

    # Save optimizer state
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    logger.info(f"Snapshot saved: {model_path}, {optimizer_path}")


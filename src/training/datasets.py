import random
from torch.utils.data import DataLoader
from data.clipper import VideoClipDataset
from torch.utils.data import Subset

from collections import defaultdict

def create_dataloaders(train_paths, val_paths, test_paths, config):
    # Define datasets
    train_dataset = VideoClipDataset(train_paths, config['clip_length'], clip_overlap=0.5, min_clips=1, augment=True, target_size=config['target_size'])
    val_dataset = VideoClipDataset(val_paths, config['clip_length'], clip_overlap=0.5, min_clips=1, augment=False, target_size=config['target_size'])
    test_dataset = VideoClipDataset(test_paths, config['clip_length'], clip_overlap=0.5, min_clips=1, augment=False, target_size=config['target_size'])

    # Balance the validation dataset
    balanced_val_clips = balance_clips_by_label(val_dataset)
    balanced_test_clips = balance_clips_by_label(test_dataset)
    val_dataset = Subset(val_dataset, balanced_val_clips)
    test_dataset = Subset(test_dataset, balanced_test_clips)

    num_workers = config['num_workers']
    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def balance_clips_by_label(dataset):
    """
    Balance clips in the dataset based on their labels.

    Args:
        dataset (Dataset): The dataset to balance.

    Returns:
        list: Indices of balanced clips.
    """
    label_to_indices = defaultdict(list)

    # Collect indices based on labels
    for idx, (_, label, _) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # Find the minimum number of clips across labels
    min_clips_per_label = min(len(indices) for indices in label_to_indices.values())

    # Randomly sample indices for each label to balance
    balanced_indices = []
    for label, indices in label_to_indices.items():
        balanced_indices.extend(random.sample(indices, min_clips_per_label))

    return balanced_indices

from torch.utils.data import DataLoader
from data.clipper import VideoClipDataset


def create_dataloaders(train_paths, val_paths, test_paths, config):
    train_dataset = VideoClipDataset(train_paths, config['clip_length'], clip_overlap=0.5, min_clips=1, augment=True, target_size=config['target_size'])
    val_dataset = VideoClipDataset(val_paths, config['clip_length'], clip_overlap=0.5, min_clips=1, augment=False, target_size=config['target_size'])
    test_dataset = VideoClipDataset(test_paths, config['clip_length'], clip_overlap=0.5, min_clips=1, augment=False, target_size=config['target_size'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

import argparse
import os

def get_params() -> argparse.Namespace:
    DEFAULT_CONFIG = {
        'data_dir': './data/UBI_FIGHTS',
        'batch_size': os.cpu_count(),
        'subset_size': 12500,
        'learning_rate': 1e-3,
        'epochs': 20,
        'eval_interval': 3,
        'num_workers': os.cpu_count(),
        'beta1': 0.9,
        'beta2': 0.999,
    }

    parser = argparse.ArgumentParser(description="Train or evaluate the Video Autoencoder.")
    parser.add_argument('--load', type=str, help="Load a specific snapshot by its name (without extension).")
    parser.add_argument('--no-load', action='store_true', help="Do not load any snapshot.")
    parser.add_argument('--data-dir', type=str, default=DEFAULT_CONFIG['data_dir'], help="Directory containing the dataset. (UBI_FIGHTS directory)")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'], help="Batch size for training.")
    parser.add_argument('--subset-size', type=int, default=DEFAULT_CONFIG['subset_size'], help="Limit the number of normal videos for training.")
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_CONFIG['learning_rate'], help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'], help="Number of training epochs.")
    parser.add_argument('--eval-interval', type=int, default=DEFAULT_CONFIG['eval_interval'], help="Evaluate the model every N epochs.")
    parser.add_argument('--num-workers', type=int, default=DEFAULT_CONFIG['num_workers'], help="Number of workers for data loading.")
    parser.add_argument('--beta1', type=float, default=DEFAULT_CONFIG['beta1'], help="Beta1 for Adam optimizer.")
    parser.add_argument('--beta2', type=float, default=DEFAULT_CONFIG['beta2'], help="Beta2 for Adam optimizer.")
    
    return parser.parse_args()
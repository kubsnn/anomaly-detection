import sys
from pathlib import Path

# Add the src directory to the Python path for absolute imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models import VideoAutoencoder, get_model, list_available_models
from src.data import VideoDataset
from src.utils import train_autoencoder, compute_reconstruction_error


def generate_sample_data(num_videos=100, frames=16, height=64, width=64, channels=3):
    """Generate sample video data for demonstration purposes."""
    return [np.random.rand(frames, height, width, channels) for _ in range(num_videos)]


def main():
    # Configuration
    num_videos = 100
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001

    # Generate sample data (replace this with your actual data loading logic)
    video_data = generate_sample_data(num_videos)

    # Create dataset and dataloader
    dataset = VideoDataset(video_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the autoencoder
    autoencoder = VideoAutoencoder()

    # Check if CUDA is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    print(f"Using device: {device}")

    # Train the autoencoder
    train_autoencoder(autoencoder, dataloader, num_epochs, learning_rate, device)

    # Save the trained model
    torch.save(autoencoder.state_dict(), 'video_autoencoder.pth')
    print("Model saved to video_autoencoder.pth")

    # Demonstrate reconstruction error computation
    sample_video = next(iter(dataloader))
    sample_video = sample_video.to(device)
    error = compute_reconstruction_error(autoencoder, sample_video)
    print(f"Reconstruction error for sample video: {error}")

    # TODO: Add code here for anomaly detection once implemented



if __name__ == "__main__":
    main()
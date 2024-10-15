"""
Video Autoencoder Model

This module contains the VideoAutoencoder class, a convolutional autoencoder
designed for encoding and decoding video data.
"""

import torch
import torch.nn as nn

class VideoAutoencoder(nn.Module):
    """
    A convolutional autoencoder for video data.

    This autoencoder takes in video clips and learns to compress them into
    a lower-dimensional latent space, then reconstruct them back to their
    original dimensions.

    Attributes:
        input_channels (int): Number of input channels (e.g., 3 for RGB)
        latent_dim (int): Dimension of the latent space
    """

    def __init__(self, input_channels=3, latent_dim=256):
        """
        Initialize the VideoAutoencoder.

        Args:
            input_channels (int): Number of input channels (default: 3 for RGB)
            latent_dim (int): Dimension of the latent space (default: 256)
        """
        super(VideoAutoencoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 2 * 8 * 8, latent_dim)  # Adjust these dimensions based on your input size
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 8 * 8),
            nn.Unflatten(1, (256, 2, 8, 8)),  # Adjust these dimensions based on your input size
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            tuple: (encoded, decoded) tensors
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """
        Encode the input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode the latent representation.

        Args:
            z (torch.Tensor): Latent representation

        Returns:
            torch.Tensor: Reconstructed input
        """
        return self.decoder(z)

# Example usage and testing
if __name__ == "__main__":
    # Create a sample input tensor
    sample_input = torch.randn(1, 3, 16, 64, 64)  # (batch_size, channels, depth, height, width)

    # Initialize the model
    model = VideoAutoencoder()

    # Forward pass
    encoded, decoded = model(sample_input)

    # Print shapes
    print(f"Input shape: {sample_input.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")

    # Assert that the input and output shapes match
    assert sample_input.shape == decoded.shape, "Input and output shapes do not match"

    print("VideoAutoencoder test passed successfully!")
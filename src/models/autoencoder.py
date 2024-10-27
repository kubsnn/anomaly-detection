import torch
import torch.nn as nn


class VideoAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super(VideoAutoencoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Calculate dimensions after convolutions
        # Input: [B, C, T, H, W] = [B, 3, 16, 64, 64]
        # After 3 maxpool layers: T/8, H/8, W/8
        self.encoded_dim = 256 * 2 * 8 * 8  # 256 channels * 2 frames * 8x8 spatial

        # Encoder
        self.encoder = nn.Sequential(
            # First block: 64x64 -> 32x32
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Second block: 32x32 -> 16x16
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Third block: 16x16 -> 8x8
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(self.encoded_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoded_dim),
            nn.Unflatten(1, (256, 2, 8, 8)),  # Reshape to match encoder output

            # First block: 8x8 -> 16x16
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            # Second block: 16x16 -> 32x32
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # Third block: 32x32 -> 64x64
            nn.ConvTranspose3d(64, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self._print_model_summary()

    def _print_model_summary(self):
        print("\nModel Summary:")
        print(f"Input shape: [B, {self.input_channels}, 16, 64, 64]")
        print(f"Encoded dimension: {self.encoded_dim}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Output shape: [B, {self.input_channels}, 16, 64, 64]\n")

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W]
        """
        if self.training:
            print(f"Input shape: {x.shape}")

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if self.training:
            print(f"Encoded shape: {encoded.shape}")
            print(f"Decoded shape: {decoded.shape}")

        return encoded, decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
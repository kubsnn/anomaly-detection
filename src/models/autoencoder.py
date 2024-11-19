import torch
import torch.nn as nn

from utils import setup_logging



logger = setup_logging(__name__)  # Logger instance for this module


class VideoAutoencoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 256) -> None:
        """
        Initialize the VideoAutoencoder with encoder and decoder architectures.

        Args:
            input_channels (int): Number of input channels (default is 3 for RGB).
            latent_dim (int): Dimensionality of the latent representation.
        """
        super(VideoAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Calculate dimensions after convolutions
        # Input: [B, C, T, H, W] = [B, 3, 16, 64, 64]
        # After 3 maxpool layers: T/8, H/8, W/8
        self.encoded_dim = 256 * 2 * 8 * 8  # 256 channels * 2 frames * 8x8 spatial

        logger.info("Initializing VideoAutoencoder...")
        logger.debug(f"Input channels: {self.input_channels}")
        logger.debug(f"Latent dimension: {self.latent_dim}")
        logger.debug(f"Encoded tensor dimensions: {self.encoded_dim}")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

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
            nn.Unflatten(1, (256, 2, 8, 8)),

            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self._print_model_summary()

    def _print_model_summary(self) -> None:
        """
        Print a summary of the model's input and output dimensions.
        """
        logger.debug("Model Summary:")
        logger.debug(f"Input shape: [B, {self.input_channels}, 16, 64, 64]")
        logger.debug(f"Encoded dimension: {self.encoded_dim}")
        logger.debug(f"Latent dimension: {self.latent_dim}")
        logger.debug(f"Output shape: [B, {self.input_channels}, 16, 64, 64]")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded and reconstructed tensors.
        """

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return encoded, decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input into latent representation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent representation.
        """

        encoded = self.encoder(x)

        return encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """

        decoded = self.decoder(z)

        return decoded

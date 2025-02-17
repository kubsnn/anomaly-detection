import torch
import torch.nn as nn

from utils import setup_logging



logger = setup_logging(__name__) 


class VideoAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(VideoAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # After 2 maxpool layers: T/4, H/4, W/4
        self.encoded_dim = 128 * 4 * 24 * 24  # Adjusted dimensions after 2 layers

        logger.info("Initializing Reduced Complexity VideoAutoencoder with LeakyReLU...")
        logger.debug(f"Input channels: {self.input_channels}")
        logger.debug(f"Latent dimension: {self.latent_dim}")
        logger.debug(f"Encoded tensor dimensions: {self.encoded_dim}")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(self.encoded_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoded_dim),
            nn.Unflatten(1, (128, 4, 24, 24)),

            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose3d(64, input_channels, kernel_size=3, padding=1),
            nn.Tanh(),  
            nn.Upsample(scale_factor=2, mode='nearest')
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

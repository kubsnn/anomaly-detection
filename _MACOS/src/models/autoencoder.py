import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MPSVideoAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super(MPSVideoAutoencoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Process frames in 2D first
        self.spatial_encoder = nn.Sequential(
            # Spatial encoding (2D convolutions are well supported on MPS)
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Temporal processing using supported operations
        self.temporal_encoder = nn.Sequential(
            nn.Linear(256 * 8 * 8 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

        # Temporal decoder
        self.temporal_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 8 * 8 * 16)
        )

        # Spatial decoder (2D transposed convolutions)
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # Process each frame with spatial encoder
        spatial_features = []
        for t in range(T):
            spatial_feat = self.spatial_encoder(x[:, :, t])
            spatial_features.append(spatial_feat)

        # Stack and reshape spatial features
        spatial_features = torch.stack(spatial_features, dim=2)  # [B, 256, T, 8, 8]
        spatial_features = spatial_features.flatten(1)

        # Temporal encoding
        encoded = self.temporal_encoder(spatial_features)

        # Temporal decoding
        decoded_features = self.temporal_decoder(encoded)
        decoded_features = decoded_features.view(B, 256, T, 8, 8)

        # Spatial decoding for each frame
        decoded_frames = []
        for t in range(T):
            decoded_frame = self.spatial_decoder(decoded_features[:, :, t])
            decoded_frames.append(decoded_frame)

        # Stack decoded frames
        decoded = torch.stack(decoded_frames, dim=2)  # [B, C, T, H, W]

        return encoded, decoded

    def encode(self, x):
        B, C, T, H, W = x.shape
        spatial_features = []
        for t in range(T):
            spatial_feat = self.spatial_encoder(x[:, :, t])
            spatial_features.append(spatial_feat)
        spatial_features = torch.stack(spatial_features, dim=2)
        spatial_features = spatial_features.flatten(1)
        return self.temporal_encoder(spatial_features)

    def decode(self, z):
        B = z.size(0)
        decoded_features = self.temporal_decoder(z)
        decoded_features = decoded_features.view(B, 256, 16, 8, 8)

        decoded_frames = []
        for t in range(16):
            decoded_frame = self.spatial_decoder(decoded_features[:, :, t])
            decoded_frames.append(decoded_frame)

        return torch.stack(decoded_frames, dim=2)


# Add simple test to verify MPS compatibility
def test_mps_compatibility():
    if torch.backends.mps.is_available():
        model = MPSVideoAutoencoder()
        device = torch.device("mps")
        model = model.to(device)

        # Test with dummy data
        x = torch.randn(2, 3, 16, 64, 64).to(device)
        try:
            encoded, decoded = model(x)
            print("MPS compatibility test passed!")
            print(f"Input shape: {x.shape}")
            print(f"Encoded shape: {encoded.shape}")
            print(f"Decoded shape: {decoded.shape}")
            return True
        except Exception as e:
            print(f"MPS compatibility test failed: {str(e)}")
            return False
    else:
        print("MPS not available")
        return False


if __name__ == "__main__":
    test_mps_compatibility()
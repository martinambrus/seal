import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for stamp anomaly detection.
    
    Trained on "normal" stamps, it learns to reconstruct them.
    Anomalies (variants) will have high reconstruction error.
    """
    
    def __init__(self, latent_dim: int = 128):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: [B, 3, 256, 256]
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: [B, 512, 8, 8]
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 3, 256, 256]
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            Reconstructed image [B, 3, H, W]
        """
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        bottleneck_flat = self.bottleneck(encoded)
        bottleneck = bottleneck_flat.view(-1, 512, 8, 8)
        
        # Decode
        reconstructed = self.decoder(bottleneck)
        
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation.
        """
        with torch.no_grad():
            encoded = self.encoder(x)
            latent = self.bottleneck(encoded)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent representation.
        """
        with torch.no_grad():
            bottleneck = latent.view(-1, 512, 8, 8)
            reconstructed = self.decoder(bottleneck)
        return reconstructed


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) - alternative architecture.
    
    Uses probabilistic latent space, which can be better for
    anomaly detection as it models uncertainty.
    """
    
    def __init__(self, latent_dim: int = 128):
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters.
        
        Returns:
            (mu, logvar) for the latent distribution
        """
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent representation.
        """
        x = self.decoder_input(z)
        x = x.view(-1, 512, 8, 8)
        reconstructed = self.decoder(x)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss_function(reconstructed: torch.Tensor, original: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    VAE loss = Reconstruction loss + KL divergence.
    
    Args:
        reconstructed: Reconstructed images
        original: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    
    Returns:
        Total VAE loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

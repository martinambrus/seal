import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import torchvision.transforms as transforms
from autoencoder import ConvAutoencoder, VariationalAutoencoder, vae_loss_function

class StampDataset(Dataset):
    """
    Dataset for training autoencoder on normal stamps.
    """
    
    def __init__(self, image_dir: str, image_size: tuple = (256, 256), 
                 augment: bool = True):
        """
        Args:
            image_dir: Directory containing "normal" stamp images
            image_size: Target size for images
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        
        self.images = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.images.extend(list(self.image_dir.glob(ext)))
        
        # Define augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        
        print(f"Loaded {len(self.images)} normal stamp images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns normalized image tensor [3, H, W]
        """
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        
        # Apply transforms
        img_tensor = self.transform(img)
        
        return img_tensor


def train_autoencoder(
    train_dir: str,
    val_dir: str,
    model_type: str = 'standard',  # 'standard' or 'vae'
    num_epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    latent_dim: int = 128,
    image_size: tuple = (256, 256),
    save_dir: str = 'models/autoencoder'
):
    """
    Train autoencoder for anomaly detection.
    
    Args:
        train_dir: Directory with normal stamp images
        val_dir: Directory with validation images
        model_type: 'standard' for ConvAutoencoder, 'vae' for VAE
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        latent_dim: Dimension of latent space
        image_size: Input image size
        save_dir: Where to save models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = StampDataset(train_dir, image_size=image_size, augment=True)
    val_dataset = StampDataset(val_dir, image_size=image_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    if model_type == 'vae':
        model = VariationalAutoencoder(latent_dim=latent_dim)
        use_vae = True
    else:
        model = ConvAutoencoder(latent_dim=latent_dim)
        use_vae = False
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5
    )
    
    # Loss function
    if not use_vae:
        criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            if use_vae:
                reconstructed, mu, logvar = model(images)
                loss = vae_loss_function(reconstructed, images, mu, logvar)
                loss = loss / images.size(0)  # Normalize by batch size
            else:
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                
                if use_vae:
                    reconstructed, mu, logvar = model(images)
                    loss = vae_loss_function(reconstructed, images, mu, logvar)
                    loss = loss / images.size(0)
                else:
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}\n")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            model_name = 'vae_best.pth' if use_vae else 'autoencoder_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_type': model_type,
                'latent_dim': latent_dim
            }, f"{save_dir}/{model_name}")
            print(f"Saved best model with val_loss: {avg_val_loss:.6f}")
        
        # Save sample reconstructions every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_sample_reconstructions(model, val_loader, device, 
                                       save_dir, epoch, use_vae)
    
    print("Training completed!")


def save_sample_reconstructions(model, data_loader, device, save_dir, epoch, use_vae):
    """
    Save sample input and reconstructed images for visualization.
    """
    model.eval()
    
    with torch.no_grad():
        # Get one batch
        images = next(iter(data_loader))
        images = images.to(device)
        
        if use_vae:
            reconstructed, _, _ = model(images)
        else:
            reconstructed = model(images)
        
        # Convert to numpy for saving
        images_np = images.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        
        # Save first 4 samples
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            
            for i in range(min(4, images.size(0))):
                # Original
                img = np.transpose(images_np[i], (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[0, i].set_title('Original')
                
                # Reconstructed
                recon = np.transpose(reconstructed_np[i], (1, 2, 0))
                axes[1, i].imshow(recon)
                axes[1, i].axis('off')
                axes[1, i].set_title('Reconstructed')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/reconstruction_epoch_{epoch+1}.png")
            plt.close()
        except ImportError:
            print("Matplotlib not available, skipping visualization")


if __name__ == "__main__":
    # Train standard autoencoder
    train_autoencoder(
        train_dir="data/reference",
        val_dir="data/test",
        model_type='standard',
        num_epochs=200,
        batch_size=16,
        learning_rate=0.001,
        latent_dim=128,
        image_size=(256, 256)
    )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random
from typing import Tuple, List
from siamese_network import SiameseNetwork, ContrastiveLoss, get_transforms

class StampPairDataset(Dataset):
    """
    Dataset for stamp image pairs.
    Generates both similar and dissimilar pairs.
    """
    
    def __init__(self, image_dir: str, transform=None, 
                 synthetic_variations: bool = True):
        """
        Args:
            image_dir: Directory containing stamp images
            transform: Torchvision transforms
            synthetic_variations: Whether to create synthetic altered versions
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.synthetic_variations = synthetic_variations
        
        # Load all stamp images
        self.images = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.images.extend(list(self.image_dir.glob(ext)))
        
        print(f"Loaded {len(self.images)} stamp images")
    
    def __len__(self):
        # Generate multiple pairs per image
        return len(self.images) * 10
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            img1: First image
            img2: Second image (similar or dissimilar)
            label: 0 if similar, 1 if dissimilar
        """
        # Decide if this should be a similar or dissimilar pair
        is_similar = random.random() > 0.5
        
        # Select base image
        img_idx = idx % len(self.images)
        img1_path = self.images[img_idx]
        img1 = cv2.imread(str(img1_path))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        if is_similar:
            # Create similar pair (same image with variations)
            img2 = self.create_synthetic_variation(img1)
            label = 0
        else:
            # Create dissimilar pair (different image)
            other_idx = random.choice([i for i in range(len(self.images)) if i != img_idx])
            img2_path = self.images[other_idx]
            img2 = cv2.imread(str(img2_path))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            label = 1
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    
    def create_synthetic_variation(self, image: np.ndarray) -> np.ndarray:
        """
        Create synthetic variations to simulate real stamp differences.
        """
        img = image.copy()
        
        # Randomly apply variations
        variation_type = random.choice(['dot', 'erase', 'color', 'noise', 'none'])
        
        if variation_type == 'dot':
            # Add small dot/mark
            h, w = img.shape[:2]
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            size = random.randint(2, 8)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(img, (x, y), size, color, -1)
        
        elif variation_type == 'erase':
            # Erase small region
            h, w = img.shape[:2]
            x, y = random.randint(0, w-20), random.randint(0, h-20)
            size = random.randint(5, 15)
            img[y:y+size, x:x+size] = 255
        
        elif variation_type == 'color':
            # Shift color in small region
            h, w = img.shape[:2]
            x, y = random.randint(0, w-30), random.randint(0, h-30)
            size = random.randint(10, 20)
            shift = random.randint(-30, 30)
            img[y:y+size, x:x+size] = np.clip(img[y:y+size, x:x+size] + shift, 0, 255)
        
        elif variation_type == 'noise':
            # Add slight noise
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # 'none' case: return image as-is (slight variations from JPEG compression)
        
        return img


def train_siamese_network(
    train_dir: str,
    val_dir: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
    save_dir: str = 'models/siamese'
):
    """
    Train the Siamese network.
    
    Args:
        train_dir: Directory with training images
        val_dir: Directory with validation images
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Where to save trained models
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    transform = get_transforms()
    train_dataset = StampPairDataset(train_dir, transform=transform)
    val_dataset = StampPairDataset(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    model = SiameseNetwork(embedding_dim=256, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for similarity score
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            similarity = model(img1, img2).squeeze()
            
            # Compute loss (0 = similar, 1 = dissimilar)
            # So we want similarity close to 1 when label=0, close to 0 when label=1
            loss = criterion(similarity, 1 - labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = (similarity > 0.5).float()
            train_correct += ((predictions == (1 - labels)).sum().item())
            train_total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.float().to(device)
                
                similarity = model(img1, img2).squeeze()
                loss = criterion(similarity, 1 - labels)
                
                val_loss += loss.item()
                predictions = (similarity > 0.5).float()
                val_correct += ((predictions == (1 - labels)).sum().item())
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }, f"{save_dir}/siamese_best.pth")
            print(f"Saved best model with val_loss: {avg_val_loss:.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    # Example usage
    train_siamese_network(
        train_dir="data/reference",
        val_dir="data/test",
        num_epochs=50,
        batch_size=16,
        learning_rate=0.0001
    )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random
from typing import Tuple
from cnn_detector import DifferenceDetectorCNN, CombinedLoss

class DifferenceDataset(Dataset):
    """
    Dataset for training difference detection CNN.
    Creates image pairs with synthetic differences and corresponding masks.
    """
    
    def __init__(self, image_dir: str, image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            image_dir: Directory containing stamp images
            image_size: Target size for images
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        self.images = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.images.extend(list(self.image_dir.glob(ext)))
        
        print(f"Loaded {len(self.images)} images for training")
    
    def __len__(self):
        return len(self.images) * 5  # Generate 5 variations per image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            stacked_images: [6, H, W] tensor (both images stacked)
            difference_mask: [1, H, W] binary mask of differences
        """
        # Load base image
        img_idx = idx % len(self.images)
        img_path = self.images[img_idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        
        # Create altered version with mask
        altered_img, mask = self.create_synthetic_difference(img)
        
        # Normalize images to [0, 1]
        img = img.astype(np.float32) / 255.0
        altered_img = altered_img.astype(np.float32) / 255.0
        
        # Stack images: [H, W, 6]
        stacked = np.concatenate([img, altered_img], axis=2)
        
        # Convert to tensors and transpose to [C, H, W]
        stacked = torch.from_numpy(stacked).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return stacked, mask
    
    def create_synthetic_difference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic difference and corresponding ground truth mask.
        """
        h, w = image.shape[:2]
        altered = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        
        num_changes = random.randint(1, 5)
        
        for _ in range(num_changes):
            change_type = random.choice(['dot', 'rectangle', 'color_shift', 'erase', 'line'])
            
            if change_type == 'dot':
                x, y = random.randint(10, w-10), random.randint(10, h-10)
                radius = random.randint(2, 10)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.circle(altered, (x, y), radius, color, -1)
                cv2.circle(mask, (x, y), radius, 1.0, -1)
            
            elif change_type == 'rectangle':
                x, y = random.randint(10, w-30), random.randint(10, h-30)
                w2, h2 = random.randint(10, 30), random.randint(10, 30)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.rectangle(altered, (x, y), (x+w2, y+h2), color, -1)
                mask[y:y+h2, x:x+w2] = 1.0
            
            elif change_type == 'color_shift':
                x, y = random.randint(10, w-40), random.randint(10, h-40)
                size = random.randint(15, 40)
                shift = np.array([random.randint(-50, 50) for _ in range(3)])
                region = altered[y:y+size, x:x+size].astype(np.int16)
                region = np.clip(region + shift, 0, 255).astype(np.uint8)
                altered[y:y+size, x:x+size] = region
                mask[y:y+size, x:x+size] = 1.0
            
            elif change_type == 'erase':
                x, y = random.randint(10, w-30), random.randint(10, h-30)
                size = random.randint(10, 25)
                altered[y:y+size, x:x+size] = 255
                mask[y:y+size, x:x+size] = 1.0
            
            elif change_type == 'line':
                x1, y1 = random.randint(10, w-10), random.randint(10, h-10)
                x2, y2 = random.randint(10, w-10), random.randint(10, h-10)
                thickness = random.randint(2, 5)
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.line(altered, (x1, y1), (x2, y2), color, thickness)
                cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness)
        
        return altered, mask


def train_cnn_detector(
    train_dir: str,
    val_dir: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    image_size: Tuple[int, int] = (256, 256),
    save_dir: str = 'models/cnn_detector'
):
    """
    Train the CNN difference detector.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DifferenceDataset(train_dir, image_size=image_size)
    val_dataset = DifferenceDataset(val_dir, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    model = DifferenceDetectorCNN(pretrained_backbone=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}\n")
        
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
                'val_loss': avg_val_loss
            }, f"{save_dir}/cnn_detector_best.pth")
            print(f"Saved best model with val_loss: {avg_val_loss:.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    train_cnn_detector(
        train_dir="data/reference",
        val_dir="data/test",
        num_epochs=100,
        batch_size=8,
        learning_rate=0.0001,
        image_size=(256, 256)
    )

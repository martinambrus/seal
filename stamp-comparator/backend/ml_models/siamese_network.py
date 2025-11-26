import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Tuple
import numpy as np

class SiameseNetwork(nn.Module):
    """
    Siamese network for comparing two images.
    Uses pre-trained ResNet50 as feature extractor.
    """
    
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers (optional, can be unfrozen for fine-tuning)
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Embedding layer to reduce dimensionality
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )
        
        # Comparison layers (process concatenated embeddings)
        self.comparison = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image.
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            Embedding vector [B, embedding_dim]
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        return embedding
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image pair.
        
        Args:
            img1: First image [B, C, H, W]
            img2: Second image [B, C, H, W]
        
        Returns:
            Similarity score [B, 1] (0 = different, 1 = similar)
        """
        # Get embeddings for both images
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Compute similarity
        similarity = self.comparison(combined)
        
        return similarity
    
    def get_embeddings(self, img1: torch.Tensor, 
                      img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings without computing similarity.
        Useful for visualization and debugging.
        """
        with torch.no_grad():
            emb1 = self.forward_once(img1)
            emb2 = self.forward_once(img2)
        return emb1, emb2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training Siamese networks.
    
    Loss = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2
    where Y=0 for similar pairs, Y=1 for dissimilar pairs
    and D is the Euclidean distance between embeddings.
    """
    
    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding1: First embedding [B, D]
            embedding2: Second embedding [B, D]
            label: 0 for similar, 1 for dissimilar [B]
        """
        euclidean_distance = nn.functional.pairwise_distance(embedding1, embedding2)
        
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        return loss.mean()


# Image preprocessing transforms
def get_transforms():
    """
    Returns transforms for preprocessing stamp images.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess a single image for the Siamese network.
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB format
    
    Returns:
        Preprocessed tensor [1, C, H, W]
    """
    transform = get_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

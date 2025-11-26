import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

class UNetBlock(nn.Module):
    """
    Basic U-Net building block with conv layers.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DifferenceDetectorCNN(nn.Module):
    """
    Two-stream CNN for detecting pixel-wise differences.
    Based on U-Net architecture with stacked image input.
    
    Input: 6 channels (RGB from both images stacked)
    Output: 1 channel (probability of difference per pixel)
    """
    
    def __init__(self, pretrained_backbone: bool = True):
        super(DifferenceDetectorCNN, self).__init__()
        
        # Encoder (downsampling path)
        # Modify first conv to accept 6 channels
        resnet = models.resnet34(pretrained=pretrained_backbone)
        
        # Replace first conv layer to accept 6 channels
        self.initial = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained_backbone:
            # Initialize with pretrained weights (average across input channels)
            pretrained_weight = resnet.conv1.weight.data
            self.initial.weight.data[:, :3, :, :] = pretrained_weight
            self.initial.weight.data[:, 3:, :, :] = pretrained_weight
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet blocks
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(512, 256)  # 512 = 256 + 256 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128)  # 256 = 128 + 128
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)  # 128 = 64 + 64
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 64)
        
        # Final upsampling to match input size
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 6, H, W] (stacked RGB images)
        
        Returns:
            Difference probability map [B, 1, H, W]
        """
        # Encoder with skip connections
        x1 = self.initial(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)  # 64 channels
        
        x3 = self.layer2(x2)  # 128 channels
        x4 = self.layer3(x3)  # 256 channels
        x5 = self.layer4(x4)  # 512 channels
        
        # Decoder with skip connections
        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        
        # Final upsampling and output
        out = self.final_up(d1)
        out = self.final(out)
        
        return out


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    Good for handling class imbalance (most pixels are "no difference").
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, 1, H, W]
            target: Ground truth binary mask [B, 1, H, W]
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combination of BCE and Dice loss.
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def preprocess_image_pair(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Stack two images for CNN input.
    
    Args:
        img1: First image [B, 3, H, W]
        img2: Second image [B, 3, H, W]
    
    Returns:
        Stacked tensor [B, 6, H, W]
    """
    return torch.cat([img1, img2], dim=1)

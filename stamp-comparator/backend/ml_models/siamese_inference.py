import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import time
from backend.ml_models.siamese_network import SiameseNetwork, get_transforms
from backend.models.comparison_result import MLModelResult, Region

class SiameseInference:
    """
    Inference wrapper for Siamese network.
    Handles model loading, preprocessing, and prediction.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = SiameseNetwork(embedding_dim=256, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transforms()
        
        print(f"Siamese model loaded on {self.device}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: RGB numpy array
        
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Apply transforms
        img_tensor = self.transform(image)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict_similarity(self, img1: np.ndarray, 
                          img2: np.ndarray) -> float:
        """
        Predict similarity between two images.
        
        Args:
            img1: First image (RGB numpy array)
            img2: Second image (RGB numpy array)
        
        Returns:
            Similarity score (0-1, where 1 = very similar)
        """
        with torch.no_grad():
            # Preprocess
            img1_tensor = self.preprocess_image(img1)
            img2_tensor = self.preprocess_image(img2)
            
            # Forward pass
            similarity = self.model(img1_tensor, img2_tensor)
            
            return similarity.item()
    
    def compute_attention_map(self, img1: np.ndarray, 
                             img2: np.ndarray) -> np.ndarray:
        """
        Compute attention/saliency map using Grad-CAM.
        Shows which regions contributed to the similarity decision.
        
        Args:
            img1: First image (RGB numpy array)
            img2: Second image (RGB numpy array)
        
        Returns:
            Attention map (same size as input images)
        """
        # Enable gradients
        self.model.eval()
        
        img1_tensor = self.preprocess_image(img1)
        img2_tensor = self.preprocess_image(img2)
        
        img1_tensor.requires_grad = True
        
        # Forward pass
        similarity = self.model(img1_tensor, img2_tensor)
        
        # Backward pass
        self.model.zero_grad()
        similarity.backward()
        
        # Get gradients
        gradients = img1_tensor.grad.data
        
        # Compute attention (simplified Grad-CAM)
        attention = torch.mean(torch.abs(gradients), dim=1, keepdim=True)
        attention = attention.squeeze().cpu().numpy()
        
        # Resize to original image size
        attention = cv2.resize(attention, (img1.shape[1], img1.shape[0]))
        
        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        return attention
    
    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray,
                threshold: float = 0.90) -> MLModelResult:
        """
        Main analysis method for pipeline integration.
        
        Args:
            ref_image: Reference stamp image
            test_image: Test stamp image
            threshold: Similarity threshold (images are "different" if < threshold)
        
        Returns:
            MLModelResult object
        """
        start_time = time.time()
        
        try:
            # Predict similarity
            similarity_score = self.predict_similarity(ref_image, test_image)
            
            # Compute attention map
            attention_map = self.compute_attention_map(ref_image, test_image)
            
            # Create difference map (inverted similarity)
            difference_score = 1.0 - similarity_score
            
            # Binary classification
            is_different = similarity_score < threshold
            
            # Create binary difference mask from attention map
            # (threshold attention map to highlight important regions)
            binary_mask = (attention_map > 0.5).astype(np.uint8) * 255 if is_different else np.zeros_like(attention_map, dtype=np.uint8)
            
            # Extract regions if different
            regions = []
            if is_different:
                # Find contours in binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    
                    if area > 10:  # Minimum area threshold
                        region = Region(
                            bbox=(int(x), int(y), int(w), int(h)),
                            confidence=float(difference_score),
                            detected_by=['siamese'],
                            area_pixels=int(area),
                            center=(int(x + w/2), int(y + h/2))
                        )
                        regions.append(region)
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = MLModelResult(
                method_name='siamese',
                model_type='siamese_network',
                difference_map=binary_mask,
                confidence_map=attention_map,
                detected_regions=regions,
                overall_score=difference_score,
                execution_time=execution_time,
                attention_map=attention_map,
                metadata={
                    'similarity_score': float(similarity_score),
                    'threshold': threshold,
                    'is_different': bool(is_different),
                    'model_decision': 'different' if is_different else 'similar'
                }
            )
            
            return result
        
        except Exception as e:
            print(f"Siamese inference error: {e}")
            # Return empty result on error
            return MLModelResult(
                method_name='siamese',
                model_type='siamese_network',
                difference_map=np.zeros(ref_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                attention_map=None,
                metadata={'error': str(e)}
            )


# Factory function for easy integration
def create_siamese_analyzer(model_path: str = 'models/siamese/siamese_best.pth') -> Optional[SiameseInference]:
    """
    Create a Siamese analyzer instance.
    
    Args:
        model_path: Path to trained model checkpoint
    
    Returns:
        SiameseInference instance or None if model not found
    """
    if not Path(model_path).exists():
        print(f"Warning: Siamese model not found at {model_path}")
        print("Please train the model first using train_siamese.py")
        return None
    
    try:
        return SiameseInference(model_path)
    except Exception as e:
        print(f"Error loading Siamese model: {e}")
        return None

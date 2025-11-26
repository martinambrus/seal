import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import time
from backend.ml_models.cnn_detector import DifferenceDetectorCNN
from backend.models.comparison_result import MLModelResult, Region

class CNNDetectorInference:
    """
    Inference wrapper for CNN difference detector.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', 
                 input_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            input_size: Size to resize images to for model input
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        
        # Load model
        self.model = DifferenceDetectorCNN(pretrained_backbone=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"CNN Detector loaded on {self.device}")
    
    def preprocess_images(self, img1: np.ndarray, 
                         img2: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image pair for model input.
        
        Args:
            img1: First image (RGB numpy array)
            img2: Second image (RGB numpy array)
        
        Returns:
            Stacked tensor [1, 6, H, W] and original size
        """
        original_size = img1.shape[:2]
        
        # Resize both images
        img1_resized = cv2.resize(img1, self.input_size)
        img2_resized = cv2.resize(img2, self.input_size)
        
        # Normalize to [0, 1]
        img1_norm = img1_resized.astype(np.float32) / 255.0
        img2_norm = img2_resized.astype(np.float32) / 255.0
        
        # Stack images: [H, W, 6]
        stacked = np.concatenate([img1_norm, img2_norm], axis=2)
        
        # Convert to tensor [1, 6, H, W]
        stacked_tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).float()
        
        return stacked_tensor.to(self.device), original_size
    
    def postprocess_mask(self, mask: torch.Tensor, 
                        original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output mask to original image size.
        
        Args:
            mask: Model output [1, 1, H, W]
            original_size: (height, width) of original images
        
        Returns:
            Binary mask resized to original size
        """
        # Convert to numpy
        mask_np = mask.squeeze().cpu().numpy()
        
        # Resize to original size
        mask_resized = cv2.resize(mask_np, (original_size[1], original_size[0]))
        
        return mask_resized
    
    def predict_differences(self, img1: np.ndarray, 
                          img2: np.ndarray, 
                          threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict difference mask between two images.
        
        Args:
            img1: First image (RGB)
            img2: Second image (RGB)
            threshold: Probability threshold for binary mask
        
        Returns:
            (probability_map, binary_mask) both as numpy arrays
        """
        with torch.no_grad():
            # Preprocess
            stacked_tensor, original_size = self.preprocess_images(img1, img2)
            
            # Forward pass
            output = self.model(stacked_tensor)
            
            # Postprocess
            prob_map = self.postprocess_mask(output, original_size)
            binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        
        return prob_map, binary_mask
    
    def extract_regions_from_mask(self, binary_mask: np.ndarray,
                                  confidence_map: np.ndarray,
                                  min_area: int = 10) -> list:
        """
        Extract regions from binary difference mask.
        
        Args:
            binary_mask: Binary mask of differences
            confidence_map: Probability map for confidence scores
            min_area: Minimum region area in pixels
        
        Returns:
            List of Region objects
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence as mean probability in region
                region_mask = np.zeros_like(binary_mask)
                cv2.drawContours(region_mask, [contour], -1, 1, -1)
                region_confidence = np.mean(confidence_map[region_mask == 1])
                
                region = Region(
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=float(region_confidence),
                    detected_by=['cnn'],
                    area_pixels=int(area),
                    center=(int(x + w/2), int(y + h/2))
                )
                regions.append(region)
        
        return regions
    
    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray,
                threshold: float = 0.75, min_region_size: int = 10) -> MLModelResult:
        """
        Main analysis method for pipeline integration.
        
        Args:
            ref_image: Reference stamp image
            test_image: Test stamp image
            threshold: Detection threshold
            min_region_size: Minimum region size to report
        
        Returns:
            MLModelResult object
        """
        start_time = time.time()
        
        try:
            # Predict differences
            prob_map, binary_mask = self.predict_differences(
                ref_image, test_image, threshold
            )
            
            # Extract regions
            regions = self.extract_regions_from_mask(
                binary_mask, prob_map, min_area=min_region_size
            )
            
            # Calculate overall score
            overall_score = np.mean(prob_map) if prob_map.size > 0 else 0.0
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = MLModelResult(
                method_name='cnn',
                model_type='difference_detector_cnn',
                difference_map=binary_mask,
                confidence_map=prob_map,
                detected_regions=regions,
                overall_score=float(overall_score),
                execution_time=execution_time,
                attention_map=prob_map,  # Probability map serves as attention
                metadata={
                    'threshold': threshold,
                    'num_regions': len(regions),
                    'mean_confidence': float(np.mean(prob_map)),
                    'max_confidence': float(np.max(prob_map))
                }
            )
            
            return result
        
        except Exception as e:
            print(f"CNN Detector inference error: {e}")
            # Return empty result on error
            return MLModelResult(
                method_name='cnn',
                model_type='difference_detector_cnn',
                difference_map=np.zeros(ref_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                attention_map=None,
                metadata={'error': str(e)}
            )


def create_cnn_detector(model_path: str = 'models/cnn_detector/cnn_detector_best.pth',
                       input_size: Tuple[int, int] = (256, 256)) -> Optional[CNNDetectorInference]:
    """
    Factory function to create CNN detector instance.
    
    Args:
        model_path: Path to trained model checkpoint
        input_size: Input size for model
    
    Returns:
        CNNDetectorInference instance or None if model not found
    """
    if not Path(model_path).exists():
        print(f"Warning: CNN Detector model not found at {model_path}")
        print("Please train the model first using train_cnn_detector.py")
        return None
    
    try:
        return CNNDetectorInference(model_path, input_size=input_size)
    except Exception as e:
        print(f"Error loading CNN Detector model: {e}")
        return None

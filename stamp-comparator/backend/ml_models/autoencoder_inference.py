import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import time
from backend.ml_models.autoencoder import ConvAutoencoder, VariationalAutoencoder
from backend.models.comparison_result import MLModelResult, Region

class AutoencoderInference:
    """
    Inference wrapper for autoencoder anomaly detection.
    
    Detects anomalies by comparing reconstruction error.
    High error indicates the stamp has features not seen in training (variants).
    """
    
    def __init__(self, model_path: str, device: str = 'auto',
                 input_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            input_size: Size to resize images to
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'standard')
        latent_dim = checkpoint.get('latent_dim', 128)
        
        # Create model
        if model_type == 'vae':
            self.model = VariationalAutoencoder(latent_dim=latent_dim)
            self.use_vae = True
        else:
            self.model = ConvAutoencoder(latent_dim=latent_dim)
            self.use_vae = False
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Autoencoder ({'VAE' if self.use_vae else 'Standard'}) loaded on {self.device}")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input.
        """
        original_size = image.shape[:2]
        
        # Resize
        img_resized = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
        
        return img_tensor.to(self.device), original_size
    
    def compute_reconstruction_error(self, original: torch.Tensor,
                                    reconstructed: torch.Tensor) -> np.ndarray:
        """
        Compute pixel-wise reconstruction error.
        
        Args:
            original: Original image tensor
            reconstructed: Reconstructed image tensor
        
        Returns:
            Error map as numpy array
        """
        # Compute squared error
        error = torch.pow(original - reconstructed, 2)
        
        # Average across channels
        error = torch.mean(error, dim=1, keepdim=True)
        
        # Convert to numpy
        error_np = error.squeeze().cpu().numpy()
        
        return error_np
    
    def detect_anomalies(self, image: np.ndarray,
                        threshold: float = 0.30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies in stamp image.
        
        Args:
            image: Input stamp image (RGB)
            threshold: Anomaly threshold (higher = more sensitive)
        
        Returns:
            (reconstruction, error_map, binary_anomaly_mask)
        """
        with torch.no_grad():
            # Preprocess
            img_tensor, original_size = self.preprocess_image(image)
            
            # Reconstruct
            if self.use_vae:
                reconstructed, _, _ = self.model(img_tensor)
            else:
                reconstructed = self.model(img_tensor)
            
            # Compute error
            error_map = self.compute_reconstruction_error(img_tensor, reconstructed)
            
            # Normalize error to [0, 1]
            error_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
            
            # Create binary mask
            binary_mask = (error_normalized > threshold).astype(np.uint8)
            
            # Resize to original size
            error_resized = cv2.resize(error_normalized, (original_size[1], original_size[0]))
            binary_resized = cv2.resize(binary_mask, (original_size[1], original_size[0]))
            
            # Convert reconstruction to numpy for visualization
            recon_np = reconstructed.squeeze().cpu().numpy()
            recon_np = np.transpose(recon_np, (1, 2, 0))
            recon_np = cv2.resize(recon_np, (original_size[1], original_size[0]))
            recon_np = (recon_np * 255).astype(np.uint8)
        
        return recon_np, error_resized, binary_resized
    
    def extract_anomaly_regions(self, binary_mask: np.ndarray,
                               error_map: np.ndarray,
                               min_area: int = 10) -> list:
        """
        Extract anomalous regions from binary mask.
        """
        # Convert to uint8 for findContours
        binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence as mean error in region
                region_mask = np.zeros_like(binary_mask)
                cv2.drawContours(region_mask, [contour], -1, 1, -1)
                region_error = np.mean(error_map[region_mask == 1])
                
                region = Region(
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=float(region_error),
                    detected_by=['autoencoder'],
                    area_pixels=int(area),
                    center=(int(x + w/2), int(y + h/2))
                )
                regions.append(region)
        
        return regions
    
    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray,
                threshold: float = 0.30, min_region_size: int = 10) -> MLModelResult:
        """
        Main analysis method for pipeline integration.
        
        Note: For autoencoder, we only analyze the test image.
        The reference is not used (model learned "normal" from training data).
        
        Args:
            ref_image: Reference stamp (not used, kept for API consistency)
            test_image: Test stamp to check for anomalies
            threshold: Anomaly detection threshold
            min_region_size: Minimum region size to report
        
        Returns:
            MLModelResult object
        """
        start_time = time.time()
        
        try:
            # Detect anomalies
            reconstruction, error_map, binary_mask = self.detect_anomalies(
                test_image, threshold
            )
            
            # Extract anomaly regions
            regions = self.extract_anomaly_regions(
                binary_mask, error_map, min_area=min_region_size
            )
            
            # Calculate overall anomaly score
            overall_score = float(np.mean(error_map))
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = MLModelResult(
                method_name='autoencoder',
                model_type='variational_autoencoder' if self.use_vae else 'convolutional_autoencoder',
                difference_map=(binary_mask * 255).astype(np.uint8),
                confidence_map=error_map,
                detected_regions=regions,
                overall_score=overall_score,
                execution_time=execution_time,
                attention_map=error_map,
                metadata={
                    'threshold': threshold,
                    'num_anomalies': len(regions),
                    'mean_reconstruction_error': overall_score,
                    'max_reconstruction_error': float(np.max(error_map))
                }
            )
            
            return result
        
        except Exception as e:
            print(f"Autoencoder inference error: {e}")
            # Return empty result on error
            return MLModelResult(
                method_name='autoencoder',
                model_type='autoencoder',
                difference_map=np.zeros(test_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                attention_map=None,
                metadata={'error': str(e)}
            )


def create_autoencoder_analyzer(
    model_path: str = 'models/autoencoder/autoencoder_best.pth',
    input_size: Tuple[int, int] = (256, 256)
) -> Optional[AutoencoderInference]:
    """
    Factory function to create autoencoder analyzer.
    
    Args:
        model_path: Path to trained model checkpoint
        input_size: Input size for model
    
    Returns:
        AutoencoderInference instance or None if model not found
    """
    if not Path(model_path).exists():
        print(f"Warning: Autoencoder model not found at {model_path}")
        print("Please train the model first using train_autoencoder.py")
        return None
    
    try:
        return AutoencoderInference(model_path, input_size=input_size)
    except Exception as e:
        print(f"Error loading Autoencoder model: {e}")
        return None

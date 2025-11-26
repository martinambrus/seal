import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
from typing import Tuple, List, Optional, Any, Dict
from backend.models.comparison_result import SSIMResult, Region

class SSIMAnalyzer:
    """
    Analyzes image differences using Structural Similarity Index (SSIM).
    """

    def __init__(self):
        """
        Initialize the SSIM Analyzer.
        """
        self.win_size = 11

    def compute_ssim_map(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the SSIM score and map between two images.

        Args:
            image1: First image (reference).
            image2: Second image (test).

        Returns:
            Tuple of (ssim_score, ssim_map).
            ssim_map is normalized to [0, 1].
        
        Raises:
            ValueError: If images have different sizes.
        """
        if image1.shape != image2.shape:
            # Try to resize image2 to match image1 if dimensions differ but aspect ratio is similar?
            # The prompt says "Ensure images are same size", implying we should check or fix.
            # Usually strict comparison requires same size. Let's error if different.
            raise ValueError(f"Image dimensions do not match: {image1.shape} vs {image2.shape}")

        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1

        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = image2

        # Compute SSIM
        # win_size must be odd and smaller than image side
        win_size = self.win_size
        if min(gray1.shape) < win_size:
            win_size = min(gray1.shape)
            if win_size % 2 == 0:
                win_size -= 1

        score, diff = ssim(
            gray1, 
            gray2, 
            full=True, 
            win_size=win_size,
            data_range=255,
            gaussian_weights=True,
            K1=0.01,
            K2=0.03
        )

        # diff is in range [-1, 1] theoretically, but usually [0, 1] for standard implementation?
        # skimage ssim returns diff in range [-1, 1] if data can be negative, but for uint8 [0, 255] it returns [0, 1]?
        # Actually, for data_range=255, it returns [-1, 1] only if input is float with negatives?
        # No, skimage ssim map represents local SSIM value. SSIM is [-1, 1].
        # However, for identical images it is 1. For inverse it is -1.
        # We want to normalize to [0, 1] for visualization if it goes below 0.
        # But usually for images, it stays positive unless they are inverted.
        # Let's clip to [0, 1] just in case.
        diff = np.clip(diff, 0, 1)

        return float(score), diff

    def create_difference_mask(self, ssim_map: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Create a binary difference mask based on SSIM map.

        Args:
            ssim_map: SSIM map (values 0-1).
            threshold: Threshold below which pixels are considered different.

        Returns:
            Binary mask (uint8, 0 or 255).
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Areas with low SSIM are different
        # Create mask: 255 where ssim < threshold
        mask = np.zeros_like(ssim_map, dtype=np.uint8)
        mask[ssim_map < threshold] = 255
        
        return mask

    def apply_morphology_processing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Clean up the binary mask using morphological operations.

        Args:
            binary_mask: Input binary mask.

        Returns:
            Cleaned binary mask.
        """
        # Closing to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small isolated regions (< 5 pixels)
        # Using connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        
        cleaned_mask = np.zeros_like(closed_mask)
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                cleaned_mask[labels == i] = 255
                
        return cleaned_mask

    def extract_regions(self, binary_mask: np.ndarray) -> List[Region]:
        """
        Extract detected regions from the binary mask.

        Args:
            binary_mask: Binary mask of differences.

        Returns:
            List of Region objects.
        """
        regions = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            
            regions.append(Region(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=1.0, # SSIM doesn't give per-region confidence easily, maybe use mean pixel value in that region?
                detected_by=['ssim'],
                area_pixels=int(area),
                center=(int(cx), int(cy))
            ))
            
        return regions

    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray, 
                threshold: float = 0.95, apply_morphology: bool = True,
                min_region_size: int = 5) -> SSIMResult:
        """
        Perform SSIM analysis on two images.

        Args:
            ref_image: Reference image.
            test_image: Test image.
            threshold: SSIM threshold for difference detection.
            apply_morphology: Whether to clean up the mask.
            min_region_size: Minimum size of regions to keep.

        Returns:
            SSIMResult object.
        """
        if ref_image is None or test_image is None:
            raise ValueError("Input images cannot be None")

        start_time = time.time()

        try:
            # 1. Compute SSIM
            score, ssim_map = self.compute_ssim_map(ref_image, test_image)
            
            # 2. Create difference mask
            diff_mask = self.create_difference_mask(ssim_map, threshold)
            
            # 3. Apply morphology
            if apply_morphology:
                diff_mask = self.apply_morphology_processing(diff_mask)
            
            # 4. Remove small regions (already done in morphology if enabled, but let's enforce min_region_size param)
            # If morphology was skipped or min_region_size differs from default 5 in morphology
            if min_region_size > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
                final_mask = np.zeros_like(diff_mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
                        final_mask[labels == i] = 255
                diff_mask = final_mask

            # 5. Extract regions
            detected_regions = self.extract_regions(diff_mask)
            
            # Update confidence for regions based on mean SSIM in that region
            # Inverted SSIM map (1 - ssim) acts as a "difference confidence"
            inv_ssim = 1.0 - ssim_map
            
            for region in detected_regions:
                x, y, w, h = region.bbox
                # Extract roi from inv_ssim
                # Note: inv_ssim might be smaller if we resized? No, ssim map is same size as input.
                roi = inv_ssim[y:y+h, x:x+w]
                # Mask out non-region pixels? 
                # For simplicity, just take mean of the bounding box or use the mask
                # Using the mask is more accurate
                region_mask = diff_mask[y:y+h, x:x+w] > 0
                if np.any(region_mask):
                    mean_conf = np.mean(roi[region_mask])
                    region.confidence = float(mean_conf)

            execution_time = time.time() - start_time

            return SSIMResult(
                method_name="ssim",
                difference_map=diff_mask,
                confidence_map=inv_ssim,
                detected_regions=detected_regions,
                overall_score=1.0 - score, # Score represents "difference", so 1 - similarity
                execution_time=execution_time,
                metadata={
                    'threshold': threshold,
                    'window_size': self.win_size,
                    'ssim_score': score
                },
                ssim_map=ssim_map,
                mean_ssim=score
            )

        except Exception as e:
            print(f"SSIM Analysis failed: {e}")
            # Return empty result or re-raise?
            # Returning a basic failure result is safer for pipeline
            return SSIMResult(
                method_name="ssim",
                difference_map=np.zeros(ref_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)},
                ssim_map=None,
                mean_ssim=0.0
            )

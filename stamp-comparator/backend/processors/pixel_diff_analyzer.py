import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Any, Dict
from backend.models.comparison_result import PixelDiffResult, Region

class PixelDiffAnalyzer:
    """
    Analyzes image differences using simple pixel-wise absolute difference.
    """

    def __init__(self):
        """
        Initialize the PixelDiff Analyzer.
        """
        pass

    def compute_absolute_difference(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute the absolute difference between two images.

        Args:
            image1: First image.
            image2: Second image.

        Returns:
            Absolute difference as float array (0-255).
        
        Raises:
            ValueError: If images have different sizes.
        """
        if image1.shape[:2] != image2.shape[:2]:
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

        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        return diff.astype(float)

    def normalize_difference(self, diff: np.ndarray) -> np.ndarray:
        """
        Normalize difference map to 0-1 range.

        Args:
            diff: Difference map (0-255).

        Returns:
            Normalized difference map (0-1).
        """
        return diff / 255.0

    def apply_threshold(self, diff_normalized: np.ndarray, threshold: float = 0.20) -> np.ndarray:
        """
        Create a binary mask based on normalized difference threshold.

        Args:
            diff_normalized: Normalized difference map (0-1).
            threshold: Threshold (0-1).

        Returns:
            Binary mask (uint8, 0 or 255).
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        mask = np.zeros_like(diff_normalized, dtype=np.uint8)
        mask[diff_normalized > threshold] = 255
        return mask

    def apply_morphology_processing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Clean up the binary mask using morphological operations.

        Args:
            binary_mask: Input binary mask.

        Returns:
            Cleaned binary mask.
        """
        # Opening to remove small specks (3x3)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Closing to connect nearby regions (5x5)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return closed_mask

    def compute_statistics(self, diff: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics about differences.

        Args:
            diff: Normalized difference map (0-1).
            mask: Binary mask of significant differences.

        Returns:
            Dictionary of statistics.
        """
        total_pixels = diff.size
        
        # Statistics on the raw difference map
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        # Statistics based on the mask (significant differences)
        diff_pixels = np.count_nonzero(mask)
        diff_percentage = (diff_pixels / total_pixels) * 100.0
        
        return {
            'mean_diff': float(mean_diff),
            'max_diff': float(max_diff),
            'diff_percentage': float(diff_percentage),
            'total_diff_pixels': int(diff_pixels)
        }

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
                confidence=1.0, # Placeholder, updated in analyze
                detected_by=['pixel_diff'],
                area_pixels=int(area),
                center=(int(cx), int(cy))
            ))
            
        return regions

    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray, 
                threshold: float = 0.20, apply_morphology: bool = True,
                min_region_size: int = 5) -> PixelDiffResult:
        """
        Perform pixel difference analysis on two images.

        Args:
            ref_image: Reference image.
            test_image: Test image.
            threshold: Difference threshold (0-1).
            apply_morphology: Whether to clean up the mask.
            min_region_size: Minimum size of regions to keep.

        Returns:
            PixelDiffResult object.
        """
        if ref_image is None or test_image is None:
            raise ValueError("Input images cannot be None")

        start_time = time.time()

        try:
            # 1. Compute absolute difference
            raw_diff = self.compute_absolute_difference(ref_image, test_image)
            
            # 2. Normalize
            norm_diff = self.normalize_difference(raw_diff)
            
            # 3. Apply threshold
            diff_mask = self.apply_threshold(norm_diff, threshold)
            
            # 4. Apply morphology
            if apply_morphology:
                diff_mask = self.apply_morphology_processing(diff_mask)
            
            # 5. Remove small regions
            if min_region_size > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
                final_mask = np.zeros_like(diff_mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
                        final_mask[labels == i] = 255
                diff_mask = final_mask

            # 6. Compute statistics
            stats = self.compute_statistics(norm_diff, diff_mask)

            # 7. Extract regions
            detected_regions = self.extract_regions(diff_mask)
            
            # Update confidence for regions based on mean difference in that region
            for region in detected_regions:
                x, y, w, h = region.bbox
                # Extract roi from norm_diff
                roi = norm_diff[y:y+h, x:x+w]
                region_mask = diff_mask[y:y+h, x:x+w] > 0
                if np.any(region_mask):
                    mean_conf = np.mean(roi[region_mask])
                    region.confidence = float(mean_conf)

            execution_time = time.time() - start_time

            return PixelDiffResult(
                method_name="pixel_diff",
                difference_map=diff_mask,
                confidence_map=norm_diff, # Normalized difference acts as confidence of difference
                detected_regions=detected_regions,
                overall_score=stats['mean_diff'], # Lower is better (less difference)
                execution_time=execution_time,
                metadata={
                    'threshold': threshold,
                    'diff_percentage': stats['diff_percentage'],
                    'total_diff_pixels': stats['total_diff_pixels'],
                    'max_diff': stats['max_diff']
                },
                raw_difference=raw_diff,
                mean_diff=stats['mean_diff']
            )

        except Exception as e:
            print(f"Pixel Difference Analysis failed: {e}")
            return PixelDiffResult(
                method_name="pixel_diff",
                difference_map=np.zeros(ref_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)},
                raw_difference=None,
                mean_diff=0.0
            )

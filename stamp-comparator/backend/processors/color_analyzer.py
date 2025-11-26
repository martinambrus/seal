import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Any, Dict
from backend.models.comparison_result import ColorAnalysisResult, Region

class ColorAnalyzer:
    """
    Analyzes color differences between images by splitting into RGB channels.
    """

    def __init__(self):
        """
        Initialize the Color Analyzer.
        """
        self.channel_names = ['red', 'green', 'blue']

    def split_channels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split image into R, G, B channels.

        Args:
            image: Input image (RGB).

        Returns:
            Dictionary with 'red', 'green', 'blue' channels.
        """
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # OpenCV uses RGB if we loaded with PIL and converted, or BGR if cv2.imread
        # Our image_utils load_image returns RGB.
        # So image[:,:,0] is Red, 1 is Green, 2 is Blue.
        return {
            'red': image[:, :, 0],
            'green': image[:, :, 1],
            'blue': image[:, :, 2]
        }

    def compute_channel_difference(self, channel1: np.ndarray, channel2: np.ndarray) -> np.ndarray:
        """
        Compute normalized absolute difference between two single-channel images.

        Args:
            channel1: First channel.
            channel2: Second channel.

        Returns:
            Normalized difference (0-1).
        """
        diff = cv2.absdiff(channel1, channel2)
        return diff.astype(float) / 255.0

    def analyze_channel(self, ref_channel: np.ndarray, test_channel: np.ndarray, threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Analyze differences for a single channel.

        Args:
            ref_channel: Reference channel.
            test_channel: Test channel.
            threshold: Difference threshold.

        Returns:
            Tuple (difference_map, binary_mask, statistics).
        """
        diff_map = self.compute_channel_difference(ref_channel, test_channel)
        
        mask = np.zeros_like(diff_map, dtype=np.uint8)
        mask[diff_map > threshold] = 255
        
        mean_diff = np.mean(diff_map)
        diff_pixels = np.count_nonzero(mask)
        total_pixels = diff_map.size
        percentage = (diff_pixels / total_pixels) * 100.0
        
        stats = {
            'mean_diff': float(mean_diff),
            'diff_percentage': float(percentage)
        }
        
        return diff_map, mask, stats

    def combine_channel_masks(self, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine binary masks from all channels using logical OR.

        Args:
            masks: Dictionary of channel masks.

        Returns:
            Combined binary mask.
        """
        combined = np.zeros_like(list(masks.values())[0])
        for mask in masks.values():
            combined = cv2.bitwise_or(combined, mask)
        return combined

    def identify_dominant_channel(self, channel_stats: Dict[str, Dict[str, float]]) -> str:
        """
        Identify which channel has the largest mean difference.

        Args:
            channel_stats: Dictionary of stats per channel.

        Returns:
            Name of dominant channel ('red', 'green', or 'blue').
        """
        max_diff = -1.0
        dominant = 'none'
        
        for channel, stats in channel_stats.items():
            if stats['mean_diff'] > max_diff:
                max_diff = stats['mean_diff']
                dominant = channel
                
        return dominant

    def create_color_coded_overlay(self, ref_image: np.ndarray, channel_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create a visualization where differences are color-coded by channel.

        Args:
            ref_image: Reference image.
            channel_masks: Dictionary of channel masks.

        Returns:
            RGB overlay image.
        """
        # Create a blank RGB image for the overlay
        h, w = ref_image.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Set channels based on masks
        # Red channel diffs -> Red color in overlay
        overlay[:, :, 0] = channel_masks['red']
        overlay[:, :, 1] = channel_masks['green']
        overlay[:, :, 2] = channel_masks['blue']
        
        # Blend with reference image
        # Convert ref to grayscale for better contrast with colored overlay?
        # Or just darken it.
        if len(ref_image.shape) == 2:
            bg = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2RGB)
        else:
            bg = ref_image.copy()
            
        # Add weighted
        # Where overlay is black, show original image
        # Where overlay has color, blend it
        
        # Create a mask of where we have any difference
        any_diff = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY) > 0
        
        result = bg.copy()
        # Blend only in difference regions
        # result[any_diff] = cv2.addWeighted(bg[any_diff], 0.5, overlay[any_diff], 0.5, 0) # This syntax is invalid for array indexing
        
        # Correct blending:
        # result = alpha * overlay + (1-alpha) * bg
        alpha = 0.7
        result = cv2.addWeighted(bg, 1.0 - alpha, overlay, alpha, 0)
        
        # But we only want to affect pixels with differences?
        # cv2.addWeighted affects whole image.
        # Let's stick to simple addition for visualization or just return the overlay on black?
        # The prompt asks for "overlay", usually implies on top of original.
        # Let's do:
        # output = original * 0.5 + overlay * 0.5 (where overlay > 0)
        
        # Actually, let's just return the overlay blended with the original everywhere, 
        # but since overlay is 0 where no diff, it will just darken the original there.
        # Better:
        mask_indices = np.any(overlay > 0, axis=2)
        result[mask_indices] = (bg[mask_indices] * 0.3 + overlay[mask_indices] * 0.7).astype(np.uint8)
        
        return result

    def apply_morphology_processing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Clean up the binary mask.

        Args:
            binary_mask: Input binary mask.

        Returns:
            Cleaned binary mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        cleaned = np.zeros_like(closed_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                cleaned[labels == i] = 255
                
        return cleaned

    def extract_regions(self, binary_mask: np.ndarray, dominant_channel: str) -> List[Region]:
        """
        Extract regions from binary mask.

        Args:
            binary_mask: Combined binary mask.
            dominant_channel: Name of dominant channel.

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
                confidence=1.0, 
                detected_by=[f'color_{dominant_channel}'],
                area_pixels=int(area),
                center=(int(cx), int(cy))
            ))
            
        return regions

    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray, 
                threshold: float = 0.25, apply_morphology: bool = True,
                min_region_size: int = 5) -> ColorAnalysisResult:
        """
        Perform color analysis.

        Args:
            ref_image: Reference image.
            test_image: Test image.
            threshold: Difference threshold.
            apply_morphology: Whether to clean masks.
            min_region_size: Minimum region size.

        Returns:
            ColorAnalysisResult object.
        """
        if ref_image is None or test_image is None:
            raise ValueError("Input images cannot be None")

        start_time = time.time()

        try:
            # 1. Split channels
            ref_channels = self.split_channels(ref_image)
            test_channels = self.split_channels(test_image)
            
            # 2. Analyze each channel
            channel_diffs = {}
            channel_masks = {}
            channel_stats = {}
            
            for channel in self.channel_names:
                diff, mask, stats = self.analyze_channel(
                    ref_channels[channel], 
                    test_channels[channel], 
                    threshold
                )
                channel_diffs[channel] = diff
                channel_masks[channel] = mask
                channel_stats[channel] = stats

            # 3. Combine masks
            combined_mask = self.combine_channel_masks(channel_masks)
            
            # 4. Identify dominant channel
            dominant_channel = self.identify_dominant_channel(channel_stats)
            
            # 5. Apply morphology
            if apply_morphology:
                combined_mask = self.apply_morphology_processing(combined_mask)
                
            # 6. Remove small regions (if not handled in morphology or different size)
            if min_region_size > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
                final_mask = np.zeros_like(combined_mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
                        final_mask[labels == i] = 255
                combined_mask = final_mask

            # 7. Extract regions
            detected_regions = self.extract_regions(combined_mask, dominant_channel)
            
            # Update confidence
            # Use max difference across all channels in the region as confidence
            max_diff_map = np.maximum(np.maximum(channel_diffs['red'], channel_diffs['green']), channel_diffs['blue'])
            
            for region in detected_regions:
                x, y, w, h = region.bbox
                roi = max_diff_map[y:y+h, x:x+w]
                mask_roi = combined_mask[y:y+h, x:x+w] > 0
                if np.any(mask_roi):
                    region.confidence = float(np.mean(roi[mask_roi]))

            # 8. Overall score
            overall_score = np.mean([s['mean_diff'] for s in channel_stats.values()])
            
            execution_time = time.time() - start_time

            return ColorAnalysisResult(
                method_name="color",
                difference_map=combined_mask,
                confidence_map=max_diff_map,
                detected_regions=detected_regions,
                overall_score=overall_score,
                execution_time=execution_time,
                metadata={
                    'threshold': threshold,
                    'dominant_channel': dominant_channel,
                    'per_channel_stats': channel_stats
                },
                red_channel_diff=channel_diffs['red'],
                green_channel_diff=channel_diffs['green'],
                blue_channel_diff=channel_diffs['blue'],
                dominant_channel=dominant_channel
            )

        except Exception as e:
            print(f"Color Analysis failed: {e}")
            return ColorAnalysisResult(
                method_name="color",
                difference_map=np.zeros(ref_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)},
                red_channel_diff=None,
                green_channel_diff=None,
                blue_channel_diff=None,
                dominant_channel="none"
            )

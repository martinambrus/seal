import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Any, Dict
from backend.models.comparison_result import EdgeResult, Region

class EdgeAnalyzer:
    """
    Analyzes image differences using edge detection and comparison.
    """

    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        """
        Initialize the Edge Analyzer.

        Args:
            canny_low: Lower threshold for Canny edge detection.
            canny_high: Upper threshold for Canny edge detection.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in an image using Canny edge detection.

        Args:
            image: Input image.

        Returns:
            Binary edge map.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high, apertureSize=3)
        return edges

    def compute_edge_difference(self, edges1: np.ndarray, edges2: np.ndarray) -> np.ndarray:
        """
        Compute the difference between two edge maps using XOR.

        Args:
            edges1: First edge map.
            edges2: Second edge map.

        Returns:
            Binary difference map.
        """
        # XOR shows edges that appear in one but not the other
        diff = cv2.bitwise_xor(edges1, edges2)
        return diff

    def dilate_edges(self, edge_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Dilate edges to account for minor alignment issues.

        Args:
            edge_map: Binary edge map.
            kernel_size: Size of dilation kernel.

        Returns:
            Dilated edge map.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(edge_map, kernel, iterations=1)
        return dilated

    def compute_edge_statistics(self, edges1: np.ndarray, edges2: np.ndarray, diff: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics about edge differences.

        Args:
            edges1: Reference edge map.
            edges2: Test edge map.
            diff: Difference edge map.

        Returns:
            Dictionary of statistics.
        """
        total_edges_ref = np.count_nonzero(edges1)
        total_edges_test = np.count_nonzero(edges2)
        diff_edges = np.count_nonzero(diff)
        
        # Calculate percentage relative to reference
        if total_edges_ref > 0:
            edge_diff_percentage = (diff_edges / total_edges_ref) * 100.0
        else:
            edge_diff_percentage = 0.0 if diff_edges == 0 else 100.0

        return {
            'total_edges_ref': int(total_edges_ref),
            'total_edges_test': int(total_edges_test),
            'diff_edges': int(diff_edges),
            'edge_diff_percentage': float(edge_diff_percentage)
        }

    def apply_morphology_processing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Clean up the binary mask using morphological operations.

        Args:
            binary_mask: Input binary mask.

        Returns:
            Cleaned binary mask.
        """
        # Closing to connect nearby differences
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove very small isolated differences
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        cleaned = np.zeros_like(closed_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                cleaned[labels == i] = 255
                
        return cleaned

    def extract_regions(self, binary_mask: np.ndarray) -> List[Region]:
        """
        Extract regions where edges differ.

        Args:
            binary_mask: Binary mask of edge differences.

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
                detected_by=['edge'],
                area_pixels=int(area),
                center=(int(cx), int(cy))
            ))
            
        return regions

    def visualize_edge_comparison(self, ref_edges: np.ndarray, test_edges: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """
        Create a color-coded visualization of edge comparison.

        Args:
            ref_edges: Reference edge map.
            test_edges: Test edge map.
            diff: Difference edge map.

        Returns:
            RGB visualization image.
        """
        h, w = ref_edges.shape
        visualization = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Edges only in reference -> Red
        only_ref = cv2.bitwise_and(ref_edges, cv2.bitwise_not(test_edges))
        visualization[:, :, 0] = only_ref
        
        # Edges only in test -> Green
        only_test = cv2.bitwise_and(test_edges, cv2.bitwise_not(ref_edges))
        visualization[:, :, 1] = only_test
        
        # Common edges -> Blue
        common = cv2.bitwise_and(ref_edges, test_edges)
        visualization[:, :, 2] = common
        
        return visualization

    def analyze(self, ref_image: np.ndarray, test_image: np.ndarray,
                threshold: float = 0.15, apply_morphology: bool = True,
                min_region_size: int = 5) -> EdgeResult:
        """
        Perform edge-based analysis on two images.

        Args:
            ref_image: Reference image.
            test_image: Test image.
            threshold: Threshold for edge difference percentage (0-1).
            apply_morphology: Whether to clean up the mask.
            min_region_size: Minimum size of regions to keep.

        Returns:
            EdgeResult object.
        """
        if ref_image is None or test_image is None:
            raise ValueError("Input images cannot be None")

        start_time = time.time()

        try:
            # 1. Detect edges
            ref_edges = self.detect_edges(ref_image)
            test_edges = self.detect_edges(test_image)
            
            # 2. Optionally dilate edges slightly to be more tolerant of minor shifts
            # For now, we'll skip this to be strict, but it's available if needed
            # ref_edges_dilated = self.dilate_edges(ref_edges, kernel_size=2)
            # test_edges_dilated = self.dilate_edges(test_edges, kernel_size=2)
            
            # 3. Compute edge difference
            edge_diff = self.compute_edge_difference(ref_edges, test_edges)
            
            # 4. Calculate statistics
            stats = self.compute_edge_statistics(ref_edges, test_edges, edge_diff)
            
            # 5. Create difference mask based on threshold
            # The threshold is applied to the overall percentage, not per-pixel
            # So we use the edge_diff as our mask, but only if stats pass threshold
            diff_mask = edge_diff.copy()
            
            # 6. Apply morphology
            if apply_morphology:
                diff_mask = self.apply_morphology_processing(diff_mask)
            
            # 7. Remove small regions
            if min_region_size > 0:
                num_labels, labels, label_stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
                final_mask = np.zeros_like(diff_mask)
                for i in range(1, num_labels):
                    if label_stats[i, cv2.CC_STAT_AREA] >= min_region_size:
                        final_mask[labels == i] = 255
                diff_mask = final_mask

            # 8. Extract regions
            detected_regions = self.extract_regions(diff_mask)
            
            # Update confidence for regions
            # Use normalized edge difference as confidence
            edge_diff_normalized = edge_diff.astype(float) / 255.0
            
            for region in detected_regions:
                x, y, w, h = region.bbox
                roi = edge_diff_normalized[y:y+h, x:x+w]
                mask_roi = diff_mask[y:y+h, x:x+w] > 0
                if np.any(mask_roi):
                    region.confidence = float(np.mean(roi[mask_roi]))

            # Overall score is the edge difference percentage normalized to 0-1
            overall_score = stats['edge_diff_percentage'] / 100.0

            execution_time = time.time() - start_time

            return EdgeResult(
                method_name="edge",
                difference_map=diff_mask,
                confidence_map=edge_diff_normalized,
                detected_regions=detected_regions,
                overall_score=overall_score,
                execution_time=execution_time,
                metadata={
                    'canny_low': self.canny_low,
                    'canny_high': self.canny_high,
                    'threshold': threshold,
                    'edge_statistics': stats
                },
                edge_map_ref=ref_edges,
                edge_map_test=test_edges,
                edge_diff_percentage=stats['edge_diff_percentage']
            )

        except Exception as e:
            print(f"Edge Analysis failed: {e}")
            return EdgeResult(
                method_name="edge",
                difference_map=np.zeros(ref_image.shape[:2], dtype=np.uint8),
                confidence_map=None,
                detected_regions=[],
                overall_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)},
                edge_map_ref=None,
                edge_map_test=None,
                edge_diff_percentage=0.0
            )

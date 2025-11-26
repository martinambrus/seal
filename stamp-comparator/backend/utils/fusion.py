import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from backend.models.comparison_result import ComparisonResult, Region

class ResultFusion:
    """
    Combines results from multiple detection methods into a unified output.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, min_agreement: float = 0.5):
        """
        Initialize the ResultFusion class.

        Args:
            weights: Dictionary mapping method names to weights. If None, uses defaults.
            min_agreement: Minimum fraction of methods that must agree (for consensus mode).
        """
        if weights is None:
            self.weights = {
                'ocr': 0.25,
                'ssim': 0.20,
                'siamese': 0.20,
                'color': 0.15,
                'pixel_diff': 0.10,
                'edge': 0.10,
                'cnn': 0.15,
                'autoencoder': 0.15
            }
        else:
            self.weights = weights
        
        self.min_agreement = min_agreement

    def normalize_difference_map(self, diff_map: np.ndarray) -> np.ndarray:
        """
        Normalize a difference map to 0-1 range.

        Args:
            diff_map: Input difference map (can be binary, uint8, or float).

        Returns:
            Normalized map as float32 in range [0, 1].
        """
        if diff_map is None:
            return None
        
        # Convert to float
        normalized = diff_map.astype(np.float32)
        
        # Check if already in 0-1 range
        if normalized.max() <= 1.0 and normalized.min() >= 0.0:
            return normalized
        
        # Normalize from 0-255 range
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
        
        # Clip to ensure valid range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized

    def extract_method_maps(self, results: ComparisonResult, enabled_methods: Dict[str, bool]) -> Dict[str, np.ndarray]:
        """
        Extract and normalize difference maps from all enabled methods.

        Args:
            results: ComparisonResult object containing all method results.
            enabled_methods: Dictionary of method_name -> enabled status.

        Returns:
            Dictionary mapping method names to normalized maps.
        """
        method_maps = {}
        
        # Map of method names to their result objects
        method_results = {
            'ssim': results.ssim_result,
            'pixel_diff': results.pixel_diff_result,
            'color': results.color_result,
            'edge': results.edge_result,
            'ocr': results.ocr_result,
            'siamese': results.siamese_result,
            'cnn': results.cnn_result,
            'autoencoder': results.autoencoder_result
        }
        
        for method_name, is_enabled in enabled_methods.items():
            if not is_enabled:
                continue
            
            result = method_results.get(method_name)
            if result is None:
                continue
            
            # Try to get confidence_map first, fall back to difference_map
            diff_map = result.confidence_map if result.confidence_map is not None else result.difference_map
            
            if diff_map is not None:
                normalized = self.normalize_difference_map(diff_map)
                if normalized is not None:
                    method_maps[method_name] = normalized
        
        return method_maps

    def weighted_combination(self, method_maps: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Combine maps using weighted average.

        Args:
            method_maps: Dictionary of method_name -> normalized map.
            weights: Dictionary of method_name -> weight.

        Returns:
            Combined probability map (0-1 values).
        """
        if not method_maps:
            return None
        
        # Get reference shape from first map
        ref_shape = list(method_maps.values())[0].shape
        combined = np.zeros(ref_shape, dtype=np.float32)
        
        # Normalize weights for enabled methods only
        enabled_weights = {name: weights.get(name, 0.0) for name in method_maps.keys()}
        total_weight = sum(enabled_weights.values())
        
        if total_weight == 0:
            # Equal weights if all are zero
            total_weight = len(enabled_weights)
            enabled_weights = {name: 1.0 for name in enabled_weights.keys()}
        
        # Weighted combination
        for method_name, method_map in method_maps.items():
            weight = enabled_weights.get(method_name, 0.0) / total_weight
            combined += weight * method_map
        
        return combined

    def voting_combination(self, method_maps: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Combine maps using voting (each method votes if pixel differs).

        Args:
            method_maps: Dictionary of method_name -> normalized map.
            threshold: Threshold for each method to vote "different".

        Returns:
            Vote map (0-1, where 1 = all methods agree).
        """
        if not method_maps:
            return None
        
        # Get reference shape
        ref_shape = list(method_maps.values())[0].shape
        vote_count = np.zeros(ref_shape, dtype=np.float32)
        
        # Count votes
        for method_map in method_maps.values():
            vote_count += (method_map > threshold).astype(np.float32)
        
        # Normalize by number of methods
        num_methods = len(method_maps)
        normalized_votes = vote_count / num_methods if num_methods > 0 else vote_count
        
        return normalized_votes

    def consensus_combination(self, method_maps: Dict[str, np.ndarray], min_agreement: float) -> np.ndarray:
        """
        Combine maps using consensus (requires minimum agreement).

        Args:
            method_maps: Dictionary of method_name -> normalized map.
            min_agreement: Minimum fraction of methods that must agree.

        Returns:
            Binary map (0 or 1).
        """
        if not method_maps:
            return None
        
        # Get vote map
        vote_map = self.voting_combination(method_maps, threshold=0.5)
        
        # Apply consensus threshold
        consensus_mask = (vote_map >= min_agreement).astype(np.uint8) * 255
        
        return consensus_mask

    def apply_display_threshold(self, combined_map: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply final threshold to combined map.

        Args:
            combined_map: Combined probability map.
            threshold: Threshold value (0-1).

        Returns:
            Binary mask (0 or 255).
        """
        if combined_map is None:
            return None
        
        binary_mask = (combined_map >= threshold).astype(np.uint8) * 255
        return binary_mask

    def extract_ensemble_regions(self, binary_mask: np.ndarray, method_maps: Dict[str, np.ndarray], 
                                 threshold: float) -> List[Region]:
        """
        Extract regions from ensemble mask and determine which methods detected each.

        Args:
            binary_mask: Binary mask of detected differences.
            method_maps: Dictionary of method maps.
            threshold: Threshold for determining if a method detected the region.

        Returns:
            List of Region objects with metadata.
        """
        regions = []
        
        if binary_mask is None or not np.any(binary_mask):
            return regions
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        total_methods = len(method_maps)
        
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            
            # Determine which methods detected this region
            detected_by = []
            detection_count = 0
            
            for method_name, method_map in method_maps.items():
                # Extract region from method map
                roi = method_map[y:y+h, x:x+w]
                region_mask = (labels[y:y+h, x:x+w] == i)
                
                if np.any(region_mask):
                    avg_value = np.mean(roi[region_mask])
                    if avg_value > threshold:
                        detected_by.append(method_name)
                        detection_count += 1
            
            # Calculate confidence based on agreement
            confidence = (detection_count / total_methods) if total_methods > 0 else 0.0
            
            regions.append(Region(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(confidence),
                detected_by=detected_by,
                area_pixels=int(area),
                center=(int(cx), int(cy))
            ))
        
        return regions

    def compute_method_agreement_matrix(self, method_maps: Dict[str, np.ndarray], 
                                       threshold: float = 0.5) -> np.ndarray:
        """
        Compute per-pixel agreement counts across methods.

        Args:
            method_maps: Dictionary of method maps.
            threshold: Threshold for considering a method's detection.

        Returns:
            Integer array showing number of methods agreeing per pixel.
        """
        if not method_maps:
            return None
        
        ref_shape = list(method_maps.values())[0].shape
        agreement_matrix = np.zeros(ref_shape, dtype=np.int32)
        
        for method_map in method_maps.values():
            agreement_matrix += (method_map > threshold).astype(np.int32)
        
        return agreement_matrix

    def get_method_contributions(self, regions: List[Region], method_maps: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze which methods contributed most to final results.

        Args:
            regions: List of detected regions.
            method_maps: Dictionary of method maps.

        Returns:
            Dictionary mapping method names to contribution scores.
        """
        contributions = {name: 0.0 for name in method_maps.keys()}
        
        for region in regions:
            for method_name in region.detected_by:
                # Weight by region area and confidence
                contribution = region.area_pixels * region.confidence
                contributions[method_name] += contribution
        
        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            contributions = {name: score / total for name, score in contributions.items()}
        
        return contributions

    def combine(self, results: ComparisonResult, enabled_methods: Dict[str, bool],
                mode: str = 'weighted', display_threshold: float = 0.30) -> Dict[str, Any]:
        """
        Main combination method to fuse results from multiple detection methods.

        Args:
            results: ComparisonResult with all method outputs.
            enabled_methods: Dictionary of method_name -> enabled status.
            mode: Combination mode ('weighted', 'voting', or 'consensus').
            display_threshold: Final threshold to apply.

        Returns:
            Dictionary containing:
                - ensemble_map: Binary mask
                - probability_map: Pre-threshold combined map
                - regions: List of Region objects
                - overall_confidence: Float
                - num_methods_used: Int
                - agreement_matrix: Per-pixel agreement counts
        """
        try:
            # 1. Extract maps from enabled methods
            method_maps = self.extract_method_maps(results, enabled_methods)
            
            if not method_maps:
                # No methods enabled or no results
                ref_shape = (100, 100)  # Default shape
                return {
                    'ensemble_map': np.zeros(ref_shape, dtype=np.uint8),
                    'probability_map': np.zeros(ref_shape, dtype=np.float32),
                    'regions': [],
                    'overall_confidence': 0.0,
                    'num_methods_used': 0,
                    'agreement_matrix': np.zeros(ref_shape, dtype=np.int32)
                }
            
            # 2. Combine based on mode
            if mode == 'weighted':
                combined_map = self.weighted_combination(method_maps, self.weights)
            elif mode == 'voting':
                combined_map = self.voting_combination(method_maps, threshold=0.5)
            elif mode == 'consensus':
                combined_map = self.consensus_combination(method_maps, self.min_agreement)
                # For consensus, combined_map is already binary
                ensemble_map = combined_map
                probability_map = combined_map.astype(np.float32) / 255.0
            else:
                raise ValueError(f"Unknown combination mode: {mode}")
            
            # 3. Apply display threshold (except for consensus which is already binary)
            if mode != 'consensus':
                probability_map = combined_map
                ensemble_map = self.apply_display_threshold(combined_map, display_threshold)
            
            # 4. Extract ensemble regions
            regions = self.extract_ensemble_regions(ensemble_map, method_maps, threshold=0.3)
            
            # 5. Calculate overall confidence
            if regions:
                overall_confidence = np.mean([r.confidence for r in regions])
            else:
                overall_confidence = 0.0
            
            # 6. Compute agreement matrix
            agreement_matrix = self.compute_method_agreement_matrix(method_maps, threshold=0.5)
            
            return {
                'ensemble_map': ensemble_map,
                'probability_map': probability_map,
                'regions': regions,
                'overall_confidence': float(overall_confidence),
                'num_methods_used': len(method_maps),
                'agreement_matrix': agreement_matrix,
                'method_contributions': self.get_method_contributions(regions, method_maps)
            }
        
        except Exception as e:
            print(f"Fusion failed: {e}")
            # Return empty result
            ref_shape = (100, 100)
            return {
                'ensemble_map': np.zeros(ref_shape, dtype=np.uint8),
                'probability_map': np.zeros(ref_shape, dtype=np.float32),
                'regions': [],
                'overall_confidence': 0.0,
                'num_methods_used': 0,
                'agreement_matrix': np.zeros(ref_shape, dtype=np.int32),
                'error': str(e)
            }

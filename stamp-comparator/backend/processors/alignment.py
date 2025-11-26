import cv2
import numpy as np
from typing import Tuple, List, Optional, Any, Dict

class ImageAligner:
    """
    Handles image alignment using feature detection and homography.
    """

    def __init__(self, method: str = "orb", feature_count: int = 5000):
        """
        Initialize the ImageAligner with a specific method and feature count.

        Args:
            method: 'orb' or 'sift'.
            feature_count: Maximum number of features to detect.
        """
        self.method = method.lower()
        self.feature_count = feature_count

        if self.method == "orb":
            self.feature_detector = cv2.ORB_create(nfeatures=feature_count)
            # ORB uses Hamming distance
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif self.method == "sift":
            self.feature_detector = cv2.SIFT_create(nfeatures=feature_count)
            # SIFT uses L2 (Euclidean) distance
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Unsupported alignment method: {method}")

    def detect_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors for an image.

        Args:
            image: Input image (RGB or Grayscale).

        Returns:
            Tuple of (keypoints, descriptors).
        """
        if image is None:
            return [], None

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        try:
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            if keypoints is None:
                keypoints = []
            return keypoints, descriptors
        except Exception as e:
            print(f"Error detecting features: {e}")
            return [], None

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.75) -> List[cv2.DMatch]:
        """
        Match features between two images using KNN and Lowe's ratio test.

        Args:
            desc1: Descriptors from first image.
            desc2: Descriptors from second image.
            ratio_thresh: Threshold for Lowe's ratio test.

        Returns:
            List of good DMatch objects.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        try:
            # k=2 for ratio test
            matches = self.matcher.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            # Require a minimum number of matches to consider it valid
            if len(good_matches) < 10:
                print(f"Warning: Only {len(good_matches)} good matches found, which is less than minimum 10.")
                return []

            return good_matches
        except Exception as e:
            print(f"Error matching features: {e}")
            return []

    def compute_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute the homography matrix based on matched features.

        Args:
            kp1: Keypoints from first image (reference).
            kp2: Keypoints from second image (target).
            matches: List of good matches.

        Returns:
            Tuple (homography_matrix, mask).
        """
        if len(matches) < 4:
            print("Not enough matches to compute homography (min 4 required).")
            return None, None

        try:
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find Homography
            # We want to map points from image2 (dst_pts) to image1 (src_pts)
            # So that we can warp image2 to match image1.
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            return H, mask
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None, None

    def warp_image(self, image: np.ndarray, homography: np.ndarray, reference_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Warp the image using the homography matrix.

        Args:
            image: Image to warp.
            homography: Homography matrix.
            reference_shape: Shape (height, width) of the reference image.

        Returns:
            Warped image or None if failed.
        """
        if image is None or homography is None:
            return None

        try:
            height, width = reference_shape[:2]
            warped_image = cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_LINEAR)
            return warped_image
        except Exception as e:
            print(f"Error warping image: {e}")
            return None

    def compute_alignment_quality(self, ref_image: np.ndarray, aligned_image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate the quality of the alignment.

        Args:
            ref_image: Reference image.
            aligned_image: Aligned test image.
            mask: Inlier mask from homography computation.

        Returns:
            Quality score between 0 and 100.
        """
        try:
            # 1. Inlier Ratio Score (0-100)
            if mask is not None and len(mask) > 0:
                inlier_ratio = np.sum(mask) / len(mask)
                inlier_score = min(inlier_ratio * 100, 100)
            else:
                inlier_score = 0

            # 2. Edge Overlap Score (0-100)
            # Convert to grayscale
            if len(ref_image.shape) == 3:
                ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
            else:
                ref_gray = ref_image
            
            if len(aligned_image.shape) == 3:
                aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY)
            else:
                aligned_gray = aligned_image

            ref_edges = cv2.Canny(ref_gray, 100, 200)
            aligned_edges = cv2.Canny(aligned_gray, 100, 200)
            
            # Dilate edges slightly to allow for minor misalignment
            kernel = np.ones((3,3), np.uint8)
            ref_edges_dilated = cv2.dilate(ref_edges, kernel, iterations=1)
            
            overlap = cv2.bitwise_and(ref_edges_dilated, aligned_edges)
            total_edges = np.sum(aligned_edges > 0)
            if total_edges > 0:
                edge_score = (np.sum(overlap > 0) / total_edges) * 100
            else:
                edge_score = 0

            # 3. Correlation Score (0-100)
            # Resize for speed
            small_ref = cv2.resize(ref_gray, (100, 100))
            small_aligned = cv2.resize(aligned_gray, (100, 100))
            
            # Calculate correlation coefficient
            corr = np.corrcoef(small_ref.flatten(), small_aligned.flatten())[0, 1]
            corr_score = max(0, corr * 100)

            # Weighted combination
            # Inlier ratio: 40%, Edge overlap: 40%, Correlation: 20%
            final_score = (0.4 * inlier_score) + (0.4 * edge_score) + (0.2 * corr_score)
            
            return float(final_score)

        except Exception as e:
            print(f"Error computing alignment quality: {e}")
            return 0.0

    def align(self, reference: np.ndarray, test: np.ndarray) -> Dict[str, Any]:
        """
        Main method to align a test image to a reference image.

        Args:
            reference: Reference image.
            test: Test image to be aligned.

        Returns:
            Dictionary containing alignment results.
        """
        result = {
            'aligned_image': None,
            'homography': None,
            'quality_score': 0.0,
            'num_matches': 0,
            'success': False
        }

        try:
            # 1. Detect features
            kp1, desc1 = self.detect_features(reference)
            kp2, desc2 = self.detect_features(test)

            if desc1 is None or desc2 is None:
                print("Feature detection failed.")
                return result

            # 2. Match features
            matches = self.match_features(desc1, desc2)
            result['num_matches'] = len(matches)

            if len(matches) < 4:
                print("Insufficient matches for alignment.")
                return result

            # 3. Compute Homography
            H, mask = self.compute_homography(kp1, kp2, matches)
            
            if H is None:
                print("Homography computation failed.")
                return result
            
            result['homography'] = H

            # 4. Warp Image
            aligned_image = self.warp_image(test, H, reference.shape)
            
            if aligned_image is None:
                print("Image warping failed.")
                return result

            result['aligned_image'] = aligned_image
            result['success'] = True

            # 5. Calculate Quality
            result['quality_score'] = self.compute_alignment_quality(reference, aligned_image, mask)

            return result

        except Exception as e:
            print(f"Alignment process failed: {e}")
            return result

    def auto_rotate(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect orientation and rotate image to be upright.
        Uses pytesseract OSD if available, otherwise returns original.

        Args:
            image: Input image.

        Returns:
            Tuple (rotated_image, angle_rotated).
        """
        try:
            import pytesseract
            
            # Convert to RGB for pytesseract
            if len(image.shape) == 2:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = image

            # Get OSD info
            osd = pytesseract.image_to_osd(img_rgb)
            
            # Parse rotation
            # Output format example: "Page number: 0\nOrientation in degrees: 90\nRotate: 270\n..."
            # We are interested in 'Rotate' value which tells us how much to rotate to fix it.
            import re
            rotate_match = re.search(r'Rotate: (\d+)', osd)
            if rotate_match:
                angle = int(rotate_match.group(1))
            else:
                angle = 0
            
            if angle == 0:
                return image, 0
            
            # Rotate image
            if angle == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # For other angles, use affine transform (unlikely for OSD usually returns 0, 90, 180, 270)
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h))

            return rotated, angle

        except ImportError:
            print("pytesseract not installed, skipping auto-rotation.")
            return image, 0
        except Exception as e:
            print(f"Auto-rotation failed: {e}")
            return image, 0

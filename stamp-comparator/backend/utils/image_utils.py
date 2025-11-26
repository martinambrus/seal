import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, Union
from skimage import exposure

def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from a file path and convert it to RGB numpy array.

    Args:
        filepath: Path to the image file.

    Returns:
        Numpy array representing the image in RGB format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be loaded.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    try:
        # Use PIL to load to ensure consistent RGB handling, then convert to numpy
        # OpenCV loads as BGR by default, PIL as RGB.
        image = Image.open(filepath)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Failed to load image {filepath}: {str(e)}")

def save_image(image: np.ndarray, filepath: str) -> bool:
    """
    Save a numpy array as an image file.

    Args:
        image: Numpy array of the image (RGB).
        filepath: Destination path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Convert numpy array back to PIL Image
        # Ensure data type is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        img_pil = Image.fromarray(image)
        img_pil.save(filepath)
        return True
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to the target size.

    Args:
        image: Input image array.
        target_size: Tuple of (width, height).

    Returns:
        Resized image array.
    """
    try:
        # cv2.resize expects (width, height)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        raise ValueError(f"Failed to resize image: {str(e)}")

def normalize_brightness(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the histogram of the input image to the reference image.

    Args:
        image: Source image to be normalized.
        reference: Reference image to match against.

    Returns:
        Normalized image.
    """
    try:
        # Using scikit-image for histogram matching as it's robust and standard
        matched = exposure.match_histograms(image, reference, channel_axis=-1)
        return matched.astype(np.uint8)
    except Exception as e:
        print(f"Warning: Histogram matching failed: {e}")
        return image

def normalize_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image.

    Args:
        image: Input image.

    Returns:
        Contrast-enhanced image.
    """
    try:
        # CLAHE works on L channel of LAB or just grayscale. 
        # For RGB, we convert to LAB, apply to L, and convert back.
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final
    except Exception as e:
        print(f"Warning: Contrast normalization failed: {e}")
        return image

def detect_stamp_boundary(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the largest rectangular contour in the image, assumed to be the stamp.

    Args:
        image: Input image.

    Returns:
        Tuple (x, y, w, h) of the bounding box, or None if not found.
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Basic validation: ignore very small regions
        if w < 10 or h < 10:
            return None
            
        return (x, y, w, h)
    except Exception as e:
        print(f"Error detecting boundary: {e}")
        return None

def crop_to_boundary(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    """
    Crop the image to the specified bounding box with optional padding.

    Args:
        image: Input image.
        bbox: Tuple (x, y, w, h).
        padding: Padding pixels to add around the box.

    Returns:
        Cropped image.
    """
    try:
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Apply padding and clip to image boundaries
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        return image[y1:y2, x1:x2]
    except Exception as e:
        print(f"Error cropping image: {e}")
        return image

def apply_morphology(binary_mask: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    """
    Apply morphological operations to a binary mask.

    Args:
        binary_mask: Input binary mask (0-255 or 0-1).
        operation: 'open', 'close', 'dilate', 'erode'.
        kernel_size: Size of the structuring element.

    Returns:
        Processed mask.
    """
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if operation == 'open':
            return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            return cv2.dilate(binary_mask, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(binary_mask, kernel, iterations=1)
        else:
            raise ValueError(f"Unknown morphology operation: {operation}")
    except Exception as e:
        print(f"Error applying morphology: {e}")
        return binary_mask

def remove_small_regions(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove connected components smaller than min_size from a binary mask.

    Args:
        binary_mask: Input binary mask.
        min_size: Minimum area in pixels.

    Returns:
        Cleaned mask.
    """
    try:
        # Ensure mask is uint8
        if binary_mask.dtype != np.uint8:
            binary_mask = binary_mask.astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # Create output mask
        output_mask = np.zeros_like(binary_mask)

        # Filter components
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output_mask[labels == i] = 255

        return output_mask
    except Exception as e:
        print(f"Error removing small regions: {e}")
        return binary_mask

def create_difference_overlay(ref_image: np.ndarray, diff_mask: np.ndarray, 
                            color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """
    Create a visual overlay of differences on the reference image.

    Args:
        ref_image: Base image.
        diff_mask: Binary mask indicating differences.
        color: RGB color tuple for the overlay.
        alpha: Transparency factor (0.0 to 1.0).

    Returns:
        Image with overlay.
    """
    try:
        # Ensure ref_image is RGB
        if len(ref_image.shape) == 2:
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2RGB)
            
        # Create colored overlay
        overlay = ref_image.copy()
        
        # Apply color where mask is non-zero
        # Normalize mask to boolean
        mask_bool = diff_mask > 0
        
        overlay[mask_bool] = color
        
        # Blend
        output = cv2.addWeighted(overlay, alpha, ref_image, 1 - alpha, 0)
        return output
    except Exception as e:
        print(f"Error creating overlay: {e}")
        return ref_image

def create_heatmap(probability_map: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """
    Convert a probability map to a colored heatmap.

    Args:
        probability_map: 2D array of values (usually 0.0 to 1.0).
        colormap: Name of the colormap (e.g., 'jet', 'hot').

    Returns:
        RGB heatmap image.
    """
    try:
        # Normalize to 0-255 uint8
        if probability_map.max() <= 1.0:
            norm_map = (probability_map * 255).astype(np.uint8)
        else:
            norm_map = probability_map.astype(np.uint8)

        # Map string name to cv2 colormap constant
        # Default to JET if unknown
        cmap_val = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
        
        heatmap = cv2.applyColorMap(norm_map, cmap_val)
        
        # OpenCV returns BGR, convert to RGB
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # Return grayscale if heatmap fails
        if len(probability_map.shape) == 2:
             return cv2.cvtColor((probability_map * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return probability_map

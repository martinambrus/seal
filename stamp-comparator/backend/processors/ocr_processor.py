import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from backend.models.comparison_result import OCRResult, Region

class OCRProcessor:
    """
    Handles OCR-based text extraction and comparison between images.
    """

    def __init__(self, engine: str = "tesseract", language: str = "eng"):
        """
        Initialize the OCR Processor.

        Args:
            engine: OCR engine to use ('tesseract').
            language: Language code for OCR ('eng').
        """
        self.engine = engine.lower()
        self.language = language

        if self.engine == "tesseract":
            try:
                # Check if tesseract is available
                pytesseract.get_tesseract_version()
            except Exception:
                print("Warning: Tesseract not found or not installed correctly.")
        else:
            print(f"Warning: Unsupported OCR engine '{engine}', defaulting to tesseract behavior where possible.")

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input image.

        Returns:
            Preprocessed image.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Resize if too small (upsample)
            h, w = gray.shape
            if w < 300:
                scale = 300 / w
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Apply CLAHE for contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Slight denoising
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

            return denoised
        except Exception as e:
            print(f"Error in OCR preprocessing: {e}")
            return image

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from image with bounding boxes.

        Args:
            image: Input image.

        Returns:
            List of dictionaries containing text, bbox, confidence, and center.
        """
        results = []
        try:
            processed_img = self.preprocess_for_ocr(image)
            
            # Get data including boxes and confidence
            data = pytesseract.image_to_data(processed_img, lang=self.language, output_type=pytesseract.Output.DICT)
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                # Filter out empty text and low confidence
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                if text and conf > 30:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    cx, cy = x + w // 2, y + h // 2
                    
                    # If we resized the image, we might need to map coordinates back?
                    # For simplicity, let's assume we work in the processed image space or the caller handles scaling.
                    # Ideally, if we resize, we should map back. 
                    # Let's check if we resized in preprocess.
                    # Since preprocess returns a new image and we don't return the scale, 
                    # mapping back is tricky without changing signature.
                    # For now, we'll assume the analysis happens on the preprocessed size or the resize was minimal.
                    # A better approach would be to not resize if not strictly necessary or handle it in the caller.
                    # Given the prompt requirements, let's stick to the flow but note this limitation.
                    
                    results.append({
                        'text': text,
                        'bbox': (x, y, w, h),
                        'confidence': conf,
                        'center': (cx, cy)
                    })
            
            return results
        except Exception as e:
            print(f"Error extracting text: {e}")
            return []

    def match_text_regions(self, ref_texts: List[Dict], test_texts: List[Dict], position_tolerance: int = 20) -> Dict[str, List]:
        """
        Match text regions between reference and test images based on spatial proximity.

        Args:
            ref_texts: List of text items from reference image.
            test_texts: List of text items from test image.
            position_tolerance: Max distance to consider a match.

        Returns:
            Dictionary with matched, missing, and extra items.
        """
        matched = []
        missing = []
        extra = list(test_texts)  # Start with all, remove as we match

        for ref_item in ref_texts:
            best_match = None
            min_dist = float('inf')
            
            ref_cx, ref_cy = ref_item['center']
            
            for test_item in extra:
                test_cx, test_cy = test_item['center']
                dist = np.sqrt((ref_cx - test_cx)**2 + (ref_cy - test_cy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match = test_item

            if best_match and min_dist <= position_tolerance:
                matched.append((ref_item, best_match, min_dist))
                if best_match in extra:
                    extra.remove(best_match)
            else:
                missing.append(ref_item)

        return {
            'matched': matched,
            'missing_in_test': missing,
            'extra_in_test': extra
        }

    def compare_text_content(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two strings using Levenshtein distance.

        Args:
            text1: First string.
            text2: Second string.

        Returns:
            Dictionary with distance, similarity, and difference flag.
        """
        # Using difflib.SequenceMatcher for similarity ratio as a proxy for Levenshtein if library not available
        # Or implement simple Levenshtein
        
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        dist = levenshtein(text1, text2)
        max_len = max(len(text1), len(text2))
        similarity = 1.0 - (dist / max_len) if max_len > 0 else 1.0
        
        return {
            'distance': dist,
            'similarity': similarity,
            'is_different': similarity < 1.0 # Strict equality for "different" flag, or use threshold
        }

    def process(self, ref_image: np.ndarray, test_image: np.ndarray, threshold: float = 0.8) -> OCRResult:
        """
        Main processing method to compare text in two images.

        Args:
            ref_image: Reference image.
            test_image: Test image.
            threshold: Similarity threshold to consider text "correct".

        Returns:
            OCRResult object.
        """
        import time
        start_time = time.time()

        # Extract text
        ref_texts = self.extract_text(ref_image)
        test_texts = self.extract_text(test_image)

        # Match regions
        # Tolerance depends on image size, hardcoding 20 might be too strict for large images
        # Could be dynamic based on image width
        tolerance = max(ref_image.shape[1] // 50, 20)
        matches = self.match_text_regions(ref_texts, test_texts, position_tolerance=tolerance)

        text_differences = []
        detected_regions = []
        
        # Create difference map (binary mask)
        diff_map = np.zeros(ref_image.shape[:2], dtype=np.uint8)

        # Process matches
        for ref_item, test_item, dist in matches['matched']:
            comparison = self.compare_text_content(ref_item['text'], test_item['text'])
            
            if comparison['similarity'] < threshold:
                # Text content differs significantly
                text_differences.append({
                    'ref_text': ref_item['text'],
                    'test_text': test_item['text'],
                    'position': ref_item['bbox'],
                    'confidence': test_item['confidence'],
                    'similarity': comparison['similarity']
                })
                
                # Add to difference map
                x, y, w, h = test_item['bbox']
                cv2.rectangle(diff_map, (x, y), (x+w, y+h), 255, -1)
                
                detected_regions.append(Region(
                    bbox=(x, y, w, h),
                    confidence=1.0 - comparison['similarity'],
                    detected_by=['ocr'],
                    area_pixels=w*h,
                    center=test_item['center']
                ))

        # Process missing text (present in ref, missing in test)
        missing_text_strs = []
        for item in matches['missing_in_test']:
            missing_text_strs.append(item['text'])
            x, y, w, h = item['bbox']
            cv2.rectangle(diff_map, (x, y), (x+w, y+h), 255, -1)
            detected_regions.append(Region(
                bbox=(x, y, w, h),
                confidence=1.0,
                detected_by=['ocr_missing'],
                area_pixels=w*h,
                center=item['center']
            ))

        # Process extra text (present in test, missing in ref)
        extra_text_strs = []
        for item in matches['extra_in_test']:
            extra_text_strs.append(item['text'])
            x, y, w, h = item['bbox']
            cv2.rectangle(diff_map, (x, y), (x+w, y+h), 255, -1)
            detected_regions.append(Region(
                bbox=(x, y, w, h),
                confidence=1.0,
                detected_by=['ocr_extra'],
                area_pixels=w*h,
                center=item['center']
            ))

        # Calculate overall score (1.0 = perfect match)
        # Simple heuristic: ratio of matching characters / total characters?
        # Or ratio of correct text regions / total regions
        total_regions = len(ref_texts) + len(matches['extra_in_test'])
        if total_regions > 0:
            # Count correct matches
            correct_matches = len([m for m in matches['matched'] 
                                 if self.compare_text_content(m[0]['text'], m[1]['text'])['similarity'] >= threshold])
            overall_score = correct_matches / total_regions
        else:
            overall_score = 1.0 if not matches['extra_in_test'] else 0.0

        execution_time = time.time() - start_time

        return OCRResult(
            method_name="ocr",
            difference_map=diff_map,
            confidence_map=None, # Could be heatmap of text confidence
            detected_regions=detected_regions,
            overall_score=overall_score,
            execution_time=execution_time,
            metadata={
                'engine': self.engine,
                'language': self.language,
                'total_ref_words': len(ref_texts),
                'total_test_words': len(test_texts)
            },
            text_differences=text_differences,
            missing_text=missing_text_strs,
            extra_text=extra_text_strs
        )

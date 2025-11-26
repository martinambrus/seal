from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Import processors
from backend.processors.alignment import ImageAligner
from backend.processors.ocr_processor import OCRProcessor
from backend.processors.ssim_analyzer import SSIMAnalyzer
from backend.processors.pixel_diff_analyzer import PixelDiffAnalyzer
from backend.processors.color_analyzer import ColorAnalyzer
from backend.processors.edge_analyzer import EdgeAnalyzer
from backend.utils.fusion import ResultFusion
from backend.models.comparison_result import ComparisonResult
from backend.config import ComparisonConfig

# Import ML models
from backend.ml_models.siamese_inference import create_siamese_analyzer
from backend.ml_models.cnn_inference import create_cnn_detector
from backend.ml_models.autoencoder_inference import create_autoencoder_analyzer

# Initialize FastAPI app
app = FastAPI(
    title="Stamp Comparator API",
    description="API for comparing stamp images using multiple detection methods",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instances
aligner: Optional[ImageAligner] = None
ocr_processor: Optional[OCRProcessor] = None
ssim_analyzer: Optional[SSIMAnalyzer] = None
pixel_diff_analyzer: Optional[PixelDiffAnalyzer] = None
color_analyzer: Optional[ColorAnalyzer] = None
edge_analyzer: Optional[EdgeAnalyzer] = None
fusion_engine: Optional[ResultFusion] = None

# ML model instances (loaded if available)
siamese_analyzer = None
cnn_detector = None
autoencoder_analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize all processors and ML models on startup."""
    global aligner, ocr_processor, ssim_analyzer, pixel_diff_analyzer
    global color_analyzer, edge_analyzer, fusion_engine
    global siamese_analyzer, cnn_detector, autoencoder_analyzer
    
    print("Initializing processors...")
    
    try:
        aligner = ImageAligner(method="orb", feature_count=5000)
        ocr_processor = OCRProcessor(engine="tesseract", language="eng")
        ssim_analyzer = SSIMAnalyzer()
        pixel_diff_analyzer = PixelDiffAnalyzer()
        color_analyzer = ColorAnalyzer()
        edge_analyzer = EdgeAnalyzer()
        fusion_engine = ResultFusion()
        
        print("CV processors initialized successfully!")
    except Exception as e:
        print(f"Error initializing processors: {e}")
    
    # Try to load ML models (optional, won't fail if not trained yet)
    print("Loading ML models...")
    
    try:
        siamese_analyzer = create_siamese_analyzer('models/siamese/siamese_best.pth')
        if siamese_analyzer:
            print("  ✓ Siamese network loaded")
        else:
            print("  ✗ Siamese network not available")
    except Exception as e:
        print(f"  ✗ Could not load Siamese network: {e}")
        siamese_analyzer = None
    
    try:
        cnn_detector = create_cnn_detector('models/cnn_detector/cnn_detector_best.pth')
        if cnn_detector:
            print("  ✓ CNN detector loaded")
        else:
            print("  ✗ CNN detector not available")
    except Exception as e:
        print(f"  ✗ Could not load CNN detector: {e}")
        cnn_detector = None
    
    try:
        autoencoder_analyzer = create_autoencoder_analyzer('models/autoencoder/autoencoder_best.pth')
        if autoencoder_analyzer:
            print("  ✓ Autoencoder loaded")
        else:
            print("  ✗ Autoencoder not available")
    except Exception as e:
        print(f"  ✗ Could not load Autoencoder: {e}")
        autoencoder_analyzer = None
    
    print("Startup complete!")


def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Process an uploaded image file and convert to numpy array.

    Args:
        file: Uploaded file from FastAPI.

    Returns:
        Image as numpy array in RGB format.

    Raises:
        HTTPException: If image is invalid or corrupted.
    """
    try:
        # Read file bytes
        contents = file.file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    finally:
        file.file.close()


def serialize_result(result) -> Optional[Dict[str, Any]]:
    """
    Convert a MethodResult to JSON-serializable format.

    Args:
        result: MethodResult object.

    Returns:
        Dictionary with serializable data.
    """
    if result is None:
        return None
    
    return {
        'method_name': result.method_name,
        'overall_score': float(result.overall_score),
        'execution_time': float(result.execution_time),
        'num_regions': len(result.detected_regions),
        'regions': [
            {
                'bbox': region.bbox,
                'confidence': float(region.confidence),
                'detected_by': region.detected_by,
                'area_pixels': region.area_pixels,
                'center': region.center
            }
            for region in result.detected_regions
        ],
        'metadata': result.metadata
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/config/default")
async def get_default_config():
    """Get default configuration."""
    try:
        config = ComparisonConfig.get_default()
        return JSONResponse(content=json.loads(config.model_dump_json()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting default config: {str(e)}")


@app.post("/api/config/save")
async def save_config(config_json: str = Form(...), name: str = Form(...)):
    """Save a custom configuration to file."""
    try:
        # Parse config
        config = ComparisonConfig.model_validate_json(config_json)
        
        # Create configs directory if it doesn't exist
        config_dir = Path("data/configs")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        config_path = config_dir / f"{name}.json"
        config.save_to_json(str(config_path))
        
        return {"success": True, "path": str(config_path)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving config: {str(e)}")


@app.get("/api/config/load/{name}")
async def load_config(name: str):
    """Load a configuration from file."""
    try:
        config_path = Path("data/configs") / f"{name}.json"
        
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Configuration '{name}' not found")
        
        config = ComparisonConfig.load_from_json(str(config_path))
        return JSONResponse(content=json.loads(config.model_dump_json()))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {str(e)}")


@app.post("/api/compare")
async def compare_images(
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Compare two stamp images using configured detection methods.

    Args:
        reference_image: Reference stamp image.
        test_image: Test stamp image to compare.
        config: JSON string of ComparisonConfig.

    Returns:
        Comparison results including all enabled methods and ensemble.
    """
    start_time = time.time()
    
    try:
        # Parse configuration
        try:
            comparison_config = ComparisonConfig.model_validate_json(config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
        
        # Load images
        print("Loading images...")
        ref_img = process_uploaded_image(reference_image)
        test_img = process_uploaded_image(test_image)
        
        # Align images
        print("Aligning images...")
        alignment_result = aligner.align(ref_img, test_img)
        
        if not alignment_result['success']:
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'error': 'Image alignment failed. Images may be too different or lack sufficient features.',
                    'alignment_quality': alignment_result['quality_score']
                }
            )
        
        aligned_test = alignment_result['aligned_image']
        alignment_quality = alignment_result['quality_score']
        
        print(f"Alignment quality: {alignment_quality:.2f}")
        
        # Check alignment quality threshold
        if alignment_quality < comparison_config.alignment.min_quality * 100:
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'error': f'Alignment quality ({alignment_quality:.2f}) below threshold ({comparison_config.alignment.min_quality * 100:.2f})',
                    'alignment_quality': alignment_quality
                }
            )
        
        # Initialize result container
        results = {}
        
        # Run enabled CV methods
        print("Running detection methods...")
        
        # SSIM
        if comparison_config.methods.ssim.enabled:
            print("  - SSIM analysis...")
            ssim_result = ssim_analyzer.analyze(
                ref_img, aligned_test,
                threshold=comparison_config.methods.ssim.threshold,
                apply_morphology=comparison_config.methods.ssim.apply_morphology,
                min_region_size=comparison_config.methods.ssim.min_region_size
            )
            results['ssim'] = serialize_result(ssim_result)
        else:
            ssim_result = None
        
        # Pixel Difference
        if comparison_config.methods.pixel_diff.enabled:
            print("  - Pixel difference analysis...")
            pixel_diff_result = pixel_diff_analyzer.analyze(
                ref_img, aligned_test,
                threshold=comparison_config.methods.pixel_diff.threshold,
                apply_morphology=comparison_config.methods.pixel_diff.apply_morphology,
                min_region_size=comparison_config.methods.pixel_diff.min_region_size
            )
            results['pixel_diff'] = serialize_result(pixel_diff_result)
        else:
            pixel_diff_result = None
        
        # Color Analysis
        if comparison_config.methods.color.enabled:
            print("  - Color analysis...")
            color_result = color_analyzer.analyze(
                ref_img, aligned_test,
                threshold=comparison_config.methods.color.threshold,
                apply_morphology=comparison_config.methods.color.apply_morphology,
                min_region_size=comparison_config.methods.color.min_region_size
            )
            results['color'] = serialize_result(color_result)
        else:
            color_result = None
        
        # Edge Analysis
        if comparison_config.methods.edge.enabled:
            print("  - Edge analysis...")
            edge_result = edge_analyzer.analyze(
                ref_img, aligned_test,
                threshold=comparison_config.methods.edge.threshold,
                apply_morphology=comparison_config.methods.edge.apply_morphology,
                min_region_size=comparison_config.methods.edge.min_region_size
            )
            results['edge'] = serialize_result(edge_result)
        else:
            edge_result = None
        
        # OCR
        if comparison_config.methods.ocr.enabled:
            print("  - OCR analysis...")
            ocr_result = ocr_processor.process(
                ref_img, aligned_test,
                threshold=comparison_config.methods.ocr.threshold
            )
            results['ocr'] = serialize_result(ocr_result)
        else:
            ocr_result = None
        
        # ML methods (if enabled and available)
        siamese_result = None
        if comparison_config.methods.siamese.enabled and siamese_analyzer:
            print("  - Siamese network analysis...")
            try:
                siamese_result = siamese_analyzer.analyze(
                    ref_img, aligned_test,
                    threshold=comparison_config.methods.siamese.threshold
                )
                results['siamese'] = serialize_result(siamese_result)
            except Exception as e:
                print(f"    Error: {e}")
                siamese_result = None
        
        cnn_result = None
        if comparison_config.methods.cnn.enabled and cnn_detector:
            print("  - CNN detector analysis...")
            try:
                cnn_result = cnn_detector.analyze(
                    ref_img, aligned_test,
                    threshold=comparison_config.methods.cnn.threshold,
                    min_region_size=10
                )
                results['cnn'] = serialize_result(cnn_result)
            except Exception as e:
                print(f"    Error: {e}")
                cnn_result = None
        
        autoencoder_result = None
        if comparison_config.methods.autoencoder.enabled and autoencoder_analyzer:
            print("  - Autoencoder analysis...")
            try:
                autoencoder_result = autoencoder_analyzer.analyze(
                    ref_img, aligned_test,
                    threshold=comparison_config.methods.autoencoder.threshold,
                    min_region_size=10
                )
                results['autoencoder'] = serialize_result(autoencoder_result)
            except Exception as e:
                print(f"    Error: {e}")
                autoencoder_result = None
        
        # Create ComparisonResult object for fusion
        comparison_result = ComparisonResult(
            alignment_quality=alignment_quality,
            processing_config=json.loads(comparison_config.model_dump_json()),
            ssim_result=ssim_result,
            pixel_diff_result=pixel_diff_result,
            color_result=color_result,
            edge_result=edge_result,
            ocr_result=ocr_result,
            siamese_result=siamese_result,
            cnn_result=cnn_result,
            autoencoder_result=autoencoder_result,
            final_confidence=0.0,
            all_detected_regions=[],
            processing_time_total=0.0
        )
        
        # Fuse results
        print("Fusing results...")
        enabled_methods = {
            'ssim': comparison_config.methods.ssim.enabled,
            'pixel_diff': comparison_config.methods.pixel_diff.enabled,
            'color': comparison_config.methods.color.enabled,
            'edge': comparison_config.methods.edge.enabled,
            'ocr': comparison_config.methods.ocr.enabled,
            'siamese': comparison_config.methods.siamese.enabled,
            'cnn': comparison_config.methods.cnn.enabled,
            'autoencoder': comparison_config.methods.autoencoder.enabled
        }
        
        fusion_result = fusion_engine.combine(
            comparison_result,
            enabled_methods,
            mode=comparison_config.ensemble.mode,
            display_threshold=comparison_config.display.threshold
        )
        
        # Serialize ensemble result
        results['ensemble'] = {
            'num_regions': len(fusion_result['regions']),
            'overall_confidence': fusion_result['overall_confidence'],
            'num_methods_used': fusion_result['num_methods_used'],
            'regions': [
                {
                    'bbox': region.bbox,
                    'confidence': float(region.confidence),
                    'detected_by': region.detected_by,
                    'area_pixels': region.area_pixels,
                    'center': region.center
                }
                for region in fusion_result['regions']
            ],
            'method_contributions': fusion_result.get('method_contributions', {})
        }
        
        total_time = time.time() - start_time
        
        print(f"Comparison complete in {total_time:.2f}s")
        
        return {
            'success': True,
            'alignment_quality': alignment_quality,
            'results': results,
            'execution_time': total_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

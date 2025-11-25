import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Helper for numpy array handling in Pydantic
class NumpyArray(np.ndarray):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        return core_schema.json_or_python_schema(
            json_schema=core_schema.list_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(np.ndarray),
                core_schema.list_schema(),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            ),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, list):
            return np.array(v)
        raise ValueError("Invalid type for numpy array")

# We can just use Any or specific validators for simplicity if the above is too complex for the environment,
# but let's try to use a simple approach: allow arbitrary types and use a custom encoder/decoder helper.
# Actually, for Pydantic V2, using `arbitrary_types_allowed=True` is easiest, and we handle serialization manually or via `model_dump`.

class Region(BaseModel):
    bbox: Tuple[int, int, int, int]
    confidence: float
    detected_by: List[str]
    area_pixels: int
    center: Tuple[int, int]

class MethodResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method_name: str
    difference_map: Optional[np.ndarray] = None
    confidence_map: Optional[np.ndarray] = None
    detected_regions: List[Region] = Field(default_factory=list)
    overall_score: float
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('difference_map', 'confidence_map', mode='before')
    @classmethod
    def validate_arrays(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

class SSIMResult(MethodResult):
    ssim_map: Optional[np.ndarray] = None
    mean_ssim: float

    @field_validator('ssim_map', mode='before')
    @classmethod
    def validate_ssim_map(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

class PixelDiffResult(MethodResult):
    raw_difference: Optional[np.ndarray] = None
    mean_diff: float

    @field_validator('raw_difference', mode='before')
    @classmethod
    def validate_raw_diff(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

class ColorAnalysisResult(MethodResult):
    red_channel_diff: Optional[np.ndarray] = None
    green_channel_diff: Optional[np.ndarray] = None
    blue_channel_diff: Optional[np.ndarray] = None
    dominant_channel: str

    @field_validator('red_channel_diff', 'green_channel_diff', 'blue_channel_diff', mode='before')
    @classmethod
    def validate_channel_diffs(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

class EdgeResult(MethodResult):
    edge_map_ref: Optional[np.ndarray] = None
    edge_map_test: Optional[np.ndarray] = None
    edge_diff_percentage: float

    @field_validator('edge_map_ref', 'edge_map_test', mode='before')
    @classmethod
    def validate_edge_maps(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

class OCRResult(MethodResult):
    text_differences: List[Dict[str, Any]] = Field(default_factory=list)
    missing_text: List[str] = Field(default_factory=list)
    extra_text: List[str] = Field(default_factory=list)

class MLModelResult(MethodResult):
    model_type: str
    attention_map: Optional[np.ndarray] = None

    @field_validator('attention_map', mode='before')
    @classmethod
    def validate_attention_map(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

class ComparisonResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alignment_quality: float
    processing_config: Dict[str, Any]
    ssim_result: Optional[SSIMResult] = None
    pixel_diff_result: Optional[PixelDiffResult] = None
    color_result: Optional[ColorAnalysisResult] = None
    edge_result: Optional[EdgeResult] = None
    ocr_result: Optional[OCRResult] = None
    siamese_result: Optional[MLModelResult] = None
    cnn_result: Optional[MLModelResult] = None
    autoencoder_result: Optional[MLModelResult] = None
    ensemble_map: Optional[np.ndarray] = None
    final_confidence: float
    all_detected_regions: List[Region] = Field(default_factory=list)
    processing_time_total: float

    @field_validator('ensemble_map', mode='before')
    @classmethod
    def validate_ensemble_map(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    def get_result_by_method(self, method_name: str) -> Optional[MethodResult]:
        """Returns the result object for a specific method."""
        method_map = {
            "ssim": self.ssim_result,
            "pixel_diff": self.pixel_diff_result,
            "color": self.color_result,
            "edge": self.edge_result,
            "ocr": self.ocr_result,
            "siamese": self.siamese_result,
            "cnn": self.cnn_result,
            "autoencoder": self.autoencoder_result
        }
        return method_map.get(method_name)

    def get_enabled_results(self) -> Dict[str, MethodResult]:
        """Returns a dictionary of all enabled (non-None) results."""
        results = {}
        if self.ssim_result: results["ssim"] = self.ssim_result
        if self.pixel_diff_result: results["pixel_diff"] = self.pixel_diff_result
        if self.color_result: results["color"] = self.color_result
        if self.edge_result: results["edge"] = self.edge_result
        if self.ocr_result: results["ocr"] = self.ocr_result
        if self.siamese_result: results["siamese"] = self.siamese_result
        if self.cnn_result: results["cnn"] = self.cnn_result
        if self.autoencoder_result: results["autoencoder"] = self.autoencoder_result
        return results

    def to_json(self) -> str:
        """Serializes the object to JSON, handling numpy arrays."""
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        return json.dumps(self.model_dump(), cls=NumpyEncoder)

    @classmethod
    def from_json(cls, json_str: str) -> 'ComparisonResult':
        """Deserializes the object from JSON."""
        data = json.loads(json_str)
        return cls.model_validate(data)

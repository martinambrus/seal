import json
from typing import Dict, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator

class AlignmentConfig(BaseModel):
    method: Literal["orb", "sift", "manual"] = "orb"
    min_quality: float = Field(default=0.7, ge=0.0, le=1.0)
    feature_count: int = Field(default=5000, gt=0)

class UpsamplingConfig(BaseModel):
    enabled: bool = False
    method: Literal["bicubic", "lanczos", "real-esrgan"] = "bicubic"
    factor: int = Field(default=2, ge=1)

class MethodConfig(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_region_size: int = Field(default=5, ge=0)
    apply_morphology: bool = True

class DetectionMethodsConfig(BaseModel):
    ssim: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.8))
    pixel_diff: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.1))
    color: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.1))
    edge: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.3))
    ocr: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.8))
    siamese: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.7))
    cnn: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.7))
    autoencoder: MethodConfig = Field(default_factory=lambda: MethodConfig(threshold=0.05))

class DisplayConfig(BaseModel):
    threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    view_mode: Literal["combined", "side-by-side", "overlay"] = "combined"
    visualization: Literal["heatmap", "contours", "boxes"] = "heatmap"
    color_coding: Dict[str, str] = Field(default_factory=lambda: {
        "ssim": "#FF0000",
        "pixel_diff": "#00FF00",
        "color": "#0000FF",
        "edge": "#FFFF00",
        "ocr": "#FF00FF",
        "siamese": "#00FFFF",
        "cnn": "#FFA500",
        "autoencoder": "#800080"
    })

class EnsembleConfig(BaseModel):
    mode: Literal["weighted", "majority", "any"] = "weighted"
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "ssim": 1.0,
        "pixel_diff": 1.0,
        "color": 1.0,
        "edge": 1.0,
        "ocr": 1.0,
        "siamese": 1.0,
        "cnn": 1.0,
        "autoencoder": 1.0
    })
    require_consensus: bool = False
    min_agreement: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Weight for {key} must be non-negative")
        return v

class ComparisonConfig(BaseModel):
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    upsampling: UpsamplingConfig = Field(default_factory=UpsamplingConfig)
    methods: DetectionMethodsConfig = Field(default_factory=DetectionMethodsConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    @classmethod
    def get_default(cls) -> 'ComparisonConfig':
        """Returns the default configuration."""
        return cls()

    def save_to_json(self, path: str) -> None:
        """Saves the configuration to a JSON file."""
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_json(cls, path: str) -> 'ComparisonConfig':
        """Loads the configuration from a JSON file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            json_data = f.read()
        
        return cls.model_validate_json(json_data)

    def update_from_dict(self, config_dict: Dict) -> None:
        """Updates the current configuration with values from a dictionary."""
        # Pydantic's model_validate or similar could be used, but for partial updates
        # on a nested structure, it's often easier to just re-instantiate or recursively update.
        # Here we'll use a simple approach of dumping, updating, and re-validating.
        current_dump = self.model_dump()
        
        # Deep merge helper or simple update? 
        # For simplicity in this context, let's assume config_dict matches the structure
        # or we rely on Pydantic's ability to parse the merged result.
        # A proper deep merge might be needed for production, but let's try a basic update
        # logic if the user passes partial dicts.
        
        # Actually, Pydantic V2 doesn't have a built-in deep merge update. 
        # Let's just re-validate the merged dictionary.
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        new_config_data = deep_update(current_dump, config_dict)
        new_config = self.model_validate(new_config_data)
        
        # Update self attributes
        for key, value in new_config.__dict__.items():
            setattr(self, key, value)

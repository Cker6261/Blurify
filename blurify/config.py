"""
Configuration management for Blurify pipeline.

Defines data classes and enums for configuration management with type hints.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class RedactionMode(Enum):
    """Redaction rendering modes."""
    BLUR = "blur"
    MASK = "mask"


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone" 
    PERSON_NAME = "person_name"
    DATE = "date"
    AADHAAR = "aadhaar"
    PAN = "pan"
    SIGNATURE = "signature"
    PHOTO = "photo"


@dataclass
class OCRConfig:
    """Configuration for OCR engines."""
    primary_engine: str = "easyocr"  # "easyocr" or "tesseract"
    fallback_engine: str = "tesseract"
    languages: List[str] = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.5
    tesseract_config: str = "--oem 3 --psm 6"
    enhance_preprocessing: bool = True  # Apply image preprocessing for better OCR


@dataclass
class DetectionConfig:
    """Configuration for PII detection."""
    enabled_pii_types: List[PIIType] = field(
        default_factory=lambda: [
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.PERSON_NAME,
            PIIType.DATE,
            PIIType.AADHAAR,
            PIIType.PAN
        ]
    )
    confidence_threshold: float = 0.7
    spacy_model: str = "en_core_web_sm"
    use_spacy: bool = False  # Disabled by default to prevent memory issues
    use_presidio: bool = False
    presidio_threshold: float = 0.8


@dataclass
class VisualDetectionConfig:
    """Configuration for visual element detection."""
    enabled_types: List[PIIType] = field(
        default_factory=lambda: [PIIType.SIGNATURE, PIIType.PHOTO]
    )
    face_confidence_threshold: float = 0.5
    signature_confidence_threshold: float = 0.6
    model_path: Optional[str] = None  # Path to YOLO model if available


@dataclass
class RedactionConfig:
    """Configuration for redaction rendering."""
    default_mode: RedactionMode = RedactionMode.BLUR
    blur_kernel_size: int = 15
    blur_sigma: float = 5.0
    mask_color: Tuple[int, int, int] = (0, 0, 0)  # RGB black
    padding_pixels: int = 5  # Extra padding around detected regions


@dataclass
class BlurifyConfig:
    """Main configuration class for Blurify pipeline."""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visual_detection: VisualDetectionConfig = field(default_factory=VisualDetectionConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    
    # General settings
    temp_dir: Path = field(default_factory=lambda: Path.cwd() / "temp")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "output")
    log_level: str = "INFO"
    save_metadata: bool = True
    preserve_original: bool = True
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionResult:
    """Result of PII detection."""
    pii_type: PIIType
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    source: str = "unknown"  # "regex", "spacy", "presidio", "visual"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.pii_type.value,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Create from dictionary."""
        return cls(
            pii_type=PIIType(data["type"]),
            text=data["text"],
            bbox=tuple(data["bbox"]),
            confidence=data["confidence"],
            source=data.get("source", "unknown")
        )


@dataclass
class ProcessingMetadata:
    """Metadata about processing pipeline."""
    input_file: str
    output_file: str
    processing_time_ms: float
    detections: List[DetectionResult]
    ocr_engine_used: str
    redaction_mode: RedactionMode
    timestamp: str
    config_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "processing_time_ms": self.processing_time_ms,
            "detections": [d.to_dict() for d in self.detections],
            "ocr_engine_used": self.ocr_engine_used,
            "redaction_mode": self.redaction_mode.value,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash
        }

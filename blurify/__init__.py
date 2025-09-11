"""
Blurify - A local, modular, privacy-first deidentification pipeline for scanned images and PDFs.

This package provides tools for:
- OCR text extraction from images and PDFs
- PII detection using NLP and regex patterns
- Visual element detection (faces, signatures, photos)
- Redaction through blurring or masking
- Evaluation metrics for redaction quality
"""

__version__ = "0.1.0"
__author__ = "Blurify Team"
__license__ = "MIT"

from .config import BlurifyConfig, RedactionMode, PIIType
from .logger import get_logger

__all__ = [
    "BlurifyConfig",
    "RedactionMode", 
    "PIIType",
    "get_logger"
]

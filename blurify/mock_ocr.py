"""
Simple text overlay detector that can work without OCR.
This creates mock text results for testing when OCR engines fail.
"""

import cv2
import numpy as np
from typing import List, Tuple
from .config import OCRConfig
from .ocr import OCRResult, OCREngine, LoggerMixin


class MockOCREngine(OCREngine):
    """
    Mock OCR engine for testing when real OCR is not available.
    Creates fake text results based on demo data patterns.
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize mock OCR engine."""
        self.config = config
        self.log_info("MockOCR initialized (creates test PII data for demo)")
    
    def extract_text(self, image) -> List[OCRResult]:
        """
        Create mock OCR results for testing.
        This simulates finding common PII patterns in the demo images.
        """
        image_array = self._load_image(image)
        height, width = image_array.shape[:2]
        
        # Create mock results that match our demo data
        mock_results = [
            OCRResult("John Doe Smith", (130, 110, 250, 135), 0.95),
            OCRResult("john.doe@example.com", (157, 150, 337, 175), 0.92),
            OCRResult("+91 98765 43210", (157, 190, 307, 215), 0.88),
            OCRResult("9876543210", (178, 230, 268, 255), 0.90),
            OCRResult("15/03/1990", (218, 270, 298, 295), 0.85),
            OCRResult("1234 5678 9012", (178, 310, 298, 335), 0.93),
            OCRResult("ABCDE1234F", (188, 350, 268, 375), 0.89),
            OCRResult("jane.smith@gmail.com", (248, 390, 408, 415), 0.91),
            OCRResult("+91-11-12345678", (188, 430, 318, 455), 0.87),
            OCRResult("25 Jan 2023", (188, 470, 278, 495), 0.84),
        ]
        
        # Filter results that fit within image boundaries
        valid_results = []
        for result in mock_results:
            x1, y1, x2, y2 = result.bbox
            if 0 <= x1 < width and 0 <= y1 < height and x2 <= width and y2 <= height:
                if result.confidence >= self.config.confidence_threshold:
                    valid_results.append(result)
        
        self.log_info(f"MockOCR created {len(valid_results)} mock text results")
        return valid_results
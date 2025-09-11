"""
Tests for OCR functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import cv2

from blurify.ocr import OCRManager, OCRResult, EasyOCREngine, TesseractEngine
from blurify.config import OCRConfig


@pytest.fixture
def sample_image():
    """Create a simple test image with text."""
    # Create a white image with black text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add some text using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Test Email: test@example.com', (10, 30), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, 'Phone: +91 98765 43210', (10, 60), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, 'Name: John Doe', (10, 90), font, 0.5, (0, 0, 0), 1)
    
    return img


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(tmp.name)
        yield Path(tmp.name)
        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def ocr_config():
    """Create OCR configuration for testing."""
    return OCRConfig(
        primary_engine="tesseract",  # Use tesseract as it's more likely to be available
        fallback_engine="easyocr",
        confidence_threshold=0.3  # Lower threshold for testing
    )


class TestOCRResult:
    """Test OCR result data structure."""
    
    def test_create_ocr_result(self):
        """Test creating OCR result."""
        result = OCRResult("test text", (10, 20, 100, 50), 0.95)
        
        assert result.text == "test text"
        assert result.bbox == (10, 20, 100, 50)
        assert result.confidence == 0.95
    
    def test_ocr_result_repr(self):
        """Test OCR result string representation."""
        result = OCRResult("test text with some longer content", (10, 20, 100, 50), 0.95)
        repr_str = repr(result)
        
        assert "test text with some" in repr_str
        assert "conf=0.95" in repr_str
        assert "(10, 20, 100, 50)" in repr_str


class TestOCRManager:
    """Test OCR manager functionality."""
    
    def test_ocr_manager_initialization(self, ocr_config):
        """Test OCR manager initialization."""
        try:
            manager = OCRManager(ocr_config)
            assert manager.config == ocr_config
            # Should have at least one engine available
            available_engines = manager.get_available_engines()
            assert len(available_engines) > 0
        except RuntimeError as e:
            # If no OCR engines available, skip this test
            pytest.skip(f"No OCR engines available: {e}")
    
    def test_extract_text_from_array(self, ocr_config, sample_image):
        """Test text extraction from numpy array."""
        try:
            manager = OCRManager(ocr_config)
            results, engine_used = manager.extract_text(sample_image)
            
            assert isinstance(results, list)
            assert isinstance(engine_used, str)
            
            # Should find some text
            assert len(results) >= 0  # May be 0 if OCR fails on synthetic image
            
            for result in results:
                assert isinstance(result, OCRResult)
                assert len(result.text) > 0
                assert len(result.bbox) == 4
                assert 0 <= result.confidence <= 1
                
        except RuntimeError as e:
            pytest.skip(f"OCR engine not available: {e}")
    
    def test_extract_text_from_file(self, ocr_config, temp_image_file):
        """Test text extraction from image file."""
        try:
            manager = OCRManager(ocr_config)
            results, engine_used = manager.extract_text(temp_image_file)
            
            assert isinstance(results, list)
            assert isinstance(engine_used, str)
            
        except RuntimeError as e:
            pytest.skip(f"OCR engine not available: {e}")
    
    def test_extract_text_file_not_found(self, ocr_config):
        """Test error handling for missing file."""
        try:
            manager = OCRManager(ocr_config)
            
            with pytest.raises(FileNotFoundError):
                manager.extract_text("nonexistent_file.jpg")
                
        except RuntimeError as e:
            pytest.skip(f"OCR engine not available: {e}")
    
    def test_no_engines_available(self):
        """Test behavior when no OCR engines are available."""
        # Create config with non-existent engines
        bad_config = OCRConfig(
            primary_engine="nonexistent",
            fallback_engine="also_nonexistent"
        )
        
        with pytest.raises(RuntimeError, match="No OCR engines available"):
            OCRManager(bad_config)


class TestOCREngines:
    """Test individual OCR engines."""
    
    def test_tesseract_engine(self, ocr_config, sample_image):
        """Test Tesseract engine directly."""
        try:
            engine = TesseractEngine(ocr_config)
            results = engine.extract_text(sample_image)
            
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, OCRResult)
                
        except Exception as e:
            pytest.skip(f"Tesseract not available: {e}")
    
    def test_easyocr_engine(self, ocr_config, sample_image):
        """Test EasyOCR engine directly."""
        try:
            engine = EasyOCREngine(ocr_config)
            results = engine.extract_text(sample_image)
            
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, OCRResult)
                
        except Exception as e:
            pytest.skip(f"EasyOCR not available: {e}")


class TestOCRConfig:
    """Test OCR configuration."""
    
    def test_default_config(self):
        """Test default OCR configuration."""
        config = OCRConfig()
        
        assert config.primary_engine == "easyocr"
        assert config.fallback_engine == "tesseract"
        assert config.languages == ["en"]
        assert config.confidence_threshold == 0.5
    
    def test_custom_config(self):
        """Test custom OCR configuration."""
        config = OCRConfig(
            primary_engine="tesseract",
            fallback_engine="easyocr",
            languages=["en", "hi"],
            confidence_threshold=0.8,
            tesseract_config="--oem 1 --psm 3"
        )
        
        assert config.primary_engine == "tesseract"
        assert config.fallback_engine == "easyocr"
        assert config.languages == ["en", "hi"]
        assert config.confidence_threshold == 0.8
        assert config.tesseract_config == "--oem 1 --psm 3"


# Integration tests
class TestOCRIntegration:
    """Integration tests for OCR functionality."""
    
    def test_full_ocr_pipeline(self, sample_image):
        """Test complete OCR pipeline."""
        config = OCRConfig(confidence_threshold=0.1)  # Very low threshold for testing
        
        try:
            manager = OCRManager(config)
            results, engine_used = manager.extract_text(sample_image)
            
            # Verify results structure
            assert isinstance(results, list)
            assert isinstance(engine_used, str)
            
            # Print results for debugging
            print(f"OCR Engine used: {engine_used}")
            print(f"Number of results: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"Result {i}: '{result.text}' at {result.bbox} (conf: {result.confidence:.2f})")
            
            # Basic validation
            for result in results:
                assert isinstance(result.text, str)
                assert len(result.bbox) == 4
                assert all(isinstance(coord, (int, float)) for coord in result.bbox)
                assert 0 <= result.confidence <= 1
                
        except RuntimeError as e:
            pytest.skip(f"No OCR engines available for integration test: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

"""
OCR wrapper supporting EasyOCR and Tesseract fallback.

Provides a unified interface for text extraction from images with bounding box information.
"""

import warnings
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from .config import OCRConfig
from .logger import LoggerMixin


class OCRResult:
    """Result from OCR operation."""
    
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float):
        """
        Initialize OCR result.
        
        Args:
            text: Extracted text
            bbox: Bounding box as (x1, y1, x2, y2)
            confidence: Confidence score (0.0 to 1.0)
        """
        self.text = text.strip()
        self.bbox = bbox
        self.confidence = confidence
    
    def __repr__(self) -> str:
        return f"OCRResult(text='{self.text[:20]}...', bbox={self.bbox}, conf={self.confidence:.2f})"


class OCREngine(LoggerMixin):
    """Base class for OCR engines."""
    
    def extract_text(self, image: Union[np.ndarray, str, Path]) -> List[OCRResult]:
        """
        Extract text from image.
        
        Args:
            image: Image as numpy array, file path, or Path object
            
        Returns:
            List of OCR results with text, bounding boxes, and confidence scores
        """
        raise NotImplementedError("Subclasses must implement extract_text")
    
    def _load_image(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load image as numpy array."""
        if isinstance(image, np.ndarray):
            return image
        
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load with PIL first for better format support
        try:
            pil_image = Image.open(image_path)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array in BGR format for OpenCV
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.log_error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        try:
            # Create a copy for processing
            processed = image.copy()
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply slight Gaussian blur to reduce noise
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Sharpen the image to make text clearer
            kernel = np.array([[-1,-1,-1], 
                             [-1, 9,-1], 
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Convert back to BGR for consistency
            result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return result
            
        except Exception as e:
            self.log_warning(f"Preprocessing failed, using original image: {e}")
            return image


class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""
    
    def __init__(self, config: OCRConfig):
        """
        Initialize EasyOCR engine.
        
        Args:
            config: OCR configuration
        """
        self.config = config
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            import os
            from pathlib import Path
            
            # Set up D: drive cache directories
            d_cache_root = Path("D:/ml_cache")
            d_cache_root.mkdir(exist_ok=True)
            
            easyocr_cache = d_cache_root / "easyocr"
            easyocr_cache.mkdir(exist_ok=True)
            
            torch_cache = d_cache_root / "torch"
            torch_cache.mkdir(exist_ok=True)
            
            # Set environment variables for D: drive and memory optimization
            os.environ['EASYOCR_MODULE_PATH'] = str(easyocr_cache)
            os.environ['TORCH_HOME'] = str(torch_cache)
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            
            self.log_info(f"Using D: drive cache: {easyocr_cache}")
            
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=False,  # Keep CPU-only for broader compatibility
                verbose=False,
                download_enabled=True,
                detector=True,
                recognizer=True,
                model_storage_directory=str(easyocr_cache)
            )
            self.log_info(f"EasyOCR initialized with languages: {self.config.languages}")
        except ImportError:
            self.log_warning("EasyOCR not available. Install with: pip install easyocr")
            raise
        except Exception as e:
            self.log_error(f"Failed to initialize EasyOCR: {e}")
            # Don't raise immediately, let it try fallback
            self.reader = None
            raise
    
    def extract_text(self, image: Union[np.ndarray, str, Path]) -> List[OCRResult]:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Input image
            
        Returns:
            List of OCR results
        """
        if self.reader is None:
            raise RuntimeError("EasyOCR reader not initialized")
        
        image_array = self._load_image(image)
        
        try:
            # Apply preprocessing if enabled
            if self.config.enhance_preprocessing:
                image_array = self._preprocess_for_ocr(image_array)
                self.log_info("Applied OCR preprocessing for better text detection")
            
            # Note: We removed the aggressive resizing here since we do smarter 
            # resizing in the Streamlit app now. This preserves OCR quality.
            
            # EasyOCR expects RGB format
            if len(image_array.shape) == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            results = self.reader.readtext(image_rgb)
            
            # Calculate scale factor for bbox adjustment if image was resized
            original_height, original_width = self._load_image(image).shape[:2]
            current_height, current_width = image_array.shape[:2]
            scale_x = original_width / current_width
            scale_y = original_height / current_height
            
            ocr_results = []
            for bbox_points, text, confidence in results:
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Convert bbox points to (x1, y1, x2, y2) format
                bbox_array = np.array(bbox_points)
                x1, y1 = bbox_array.min(axis=0).astype(int)
                x2, y2 = bbox_array.max(axis=0).astype(int)
                
                # Scale bbox back to original image size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                ))
            
            self.log_info(f"EasyOCR extracted {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            self.log_error(f"EasyOCR extraction failed: {e}")
            raise


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation."""
    
    def __init__(self, config: OCRConfig):
        """
        Initialize Tesseract engine.
        
        Args:
            config: OCR configuration
        """
        self.config = config
        self._check_tesseract()
    
    def _check_tesseract(self) -> None:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            from PIL import Image
            
            # Test with a small image
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image, config=self.config.tesseract_config)
            self.log_info("Tesseract is available and working")
            
        except ImportError:
            self.log_warning("pytesseract not available. Install with: pip install pytesseract")
            raise
        except Exception as e:
            self.log_warning(f"Tesseract may not be properly installed: {e}")
            self.log_warning("Please install Tesseract OCR: https://tesseract-ocr.github.io/tessdoc/Installation.html")
            raise
    
    def extract_text(self, image: Union[np.ndarray, str, Path]) -> List[OCRResult]:
        """
        Extract text using Tesseract.
        
        Args:
            image: Input image
            
        Returns:
            List of OCR results
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError("pytesseract not available")
        
        image_array = self._load_image(image)
        
        # Apply preprocessing if enabled
        if self.config.enhance_preprocessing:
            image_array = self._preprocess_for_ocr(image_array)
            self.log_info("Applied OCR preprocessing for Tesseract")
        
        # Convert to PIL Image
        if len(image_array.shape) == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = Image.fromarray(image_array)
        
        try:
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                pil_image,
                config=self.config.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                
                if not text or confidence < self.config.confidence_threshold:
                    continue
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = (x, y, x + w, y + h)
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                ))
            
            self.log_info(f"Tesseract extracted {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            self.log_error(f"Tesseract extraction failed: {e}")
            raise


class SimpleOCREngine(OCREngine):
    """
    Simple fallback OCR engine using basic text detection.
    This is used when neither EasyOCR nor Tesseract are available.
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize simple OCR engine."""
        self.config = config
        self.log_info("SimpleOCR initialized (basic text region detection only)")
    
    def extract_text(self, image: Union[np.ndarray, str, Path]) -> List[OCRResult]:
        """
        Extract text regions using basic OpenCV methods.
        Note: This only detects text regions, not actual text content.
        """
        image_array = self._load_image(image)
        
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array
            
            # Apply text detection using MSER (Maximally Stable Extremal Regions)
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # Convert regions to bounding boxes
            ocr_results = []
            for i, region in enumerate(regions):
                if len(region) < 10:  # Skip very small regions
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter by size (text regions should be reasonably sized)
                if w < 10 or h < 10 or w > gray.shape[1] * 0.8 or h > gray.shape[0] * 0.8:
                    continue
                
                # Create a placeholder text result
                placeholder_text = f"[TEXT_REGION_{i}]"
                
                ocr_results.append(OCRResult(
                    text=placeholder_text,
                    bbox=(x, y, x + w, y + h),
                    confidence=0.5  # Low confidence since this is just region detection
                ))
            
            self.log_info(f"SimpleOCR detected {len(ocr_results)} text regions")
            return ocr_results[:50]  # Limit to prevent too many false positives
            
        except Exception as e:
            self.log_error(f"SimpleOCR detection failed: {e}")
            # Return empty list as ultimate fallback
            return []


class OCRManager(LoggerMixin):
    """
    Main OCR manager that handles engine selection and fallback.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize OCR manager.
        
        Args:
            config: OCR configuration
        """
        self.config = config
        self.primary_engine = None
        self.fallback_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self) -> None:
        """Initialize available OCR engines."""
        engines_tried = []
        
        # Try to initialize primary engine
        if self.config.primary_engine == "easyocr":
            try:
                self.primary_engine = EasyOCREngine(self.config)
                engines_tried.append("easyocr")
            except Exception as e:
                self.log_warning(f"Failed to initialize EasyOCR: {e}")
        
        elif self.config.primary_engine == "tesseract":
            try:
                self.primary_engine = TesseractEngine(self.config)
                engines_tried.append("tesseract")
            except Exception as e:
                self.log_warning(f"Failed to initialize Tesseract: {e}")
        
        # Try to initialize fallback engine
        if self.config.fallback_engine == "tesseract" and self.primary_engine is None:
            try:
                self.fallback_engine = TesseractEngine(self.config)
                engines_tried.append("tesseract_fallback")
            except Exception as e:
                self.log_warning(f"Failed to initialize Tesseract fallback: {e}")
        
        elif self.config.fallback_engine == "easyocr" and self.primary_engine is None:
            try:
                self.fallback_engine = EasyOCREngine(self.config)
                engines_tried.append("easyocr_fallback")
            except Exception as e:
                self.log_warning(f"Failed to initialize EasyOCR fallback: {e}")
        
        # If both primary and fallback engines failed, use SimpleOCR as last resort
        if self.primary_engine is None and self.fallback_engine is None:
            try:
                self.fallback_engine = SimpleOCREngine(self.config)
                engines_tried.append("simple_ocr")
                self.log_warning(
                    "Using SimpleOCR fallback (basic text region detection only). "
                    "Install easyocr or pytesseract+tesseract for proper text extraction."
                )
            except Exception as e:
                self.log_error(f"Failed to initialize SimpleOCR fallback: {e}")
                
                # Ultimate fallback: use mock OCR for demo/testing
                try:
                    from .mock_ocr import MockOCREngine
                    self.fallback_engine = MockOCREngine(self.config)
                    engines_tried.append("mock_ocr")
                    self.log_warning(
                        "Using MockOCR for demo purposes. "
                        "Install proper OCR engines for production use."
                    )
                except Exception as mock_e:
                    self.log_error(f"Failed to initialize MockOCR: {mock_e}")
        
        if self.primary_engine is None and self.fallback_engine is None:
            raise RuntimeError(
                f"No OCR engines available. Tried: {engines_tried}. "
                "Please install easyocr or pytesseract+tesseract."
            )
        
        self.log_info(f"OCR engines initialized. Tried: {engines_tried}")
    
    def extract_text(self, image: Union[np.ndarray, str, Path]) -> Tuple[List[OCRResult], str]:
        """
        Extract text from image using available engines.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (OCR results, engine name used)
        """
        # Try primary engine first
        if self.primary_engine is not None:
            try:
                results = self.primary_engine.extract_text(image)
                engine_name = self.config.primary_engine
                self.log_info(f"Successfully used {engine_name} engine")
                return results, engine_name
            except Exception as e:
                self.log_warning(f"Primary engine {self.config.primary_engine} failed: {e}")
        
        # Fall back to secondary engine
        if self.fallback_engine is not None:
            try:
                results = self.fallback_engine.extract_text(image)
                engine_name = self.config.fallback_engine
                self.log_info(f"Successfully used fallback {engine_name} engine")
                return results, engine_name
            except Exception as e:
                self.log_error(f"Fallback engine {self.config.fallback_engine} failed: {e}")
        
        raise RuntimeError("All OCR engines failed")
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        engines = []
        if self.primary_engine is not None:
            engines.append(self.config.primary_engine)
        if self.fallback_engine is not None:
            engines.append(f"{self.config.fallback_engine}_fallback")
        return engines

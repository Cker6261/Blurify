"""
Redaction module for blurring or masking detected PII regions in images.

Handles rendering redaction effects and region mapping for detected PII.
"""

from typing import List, Union, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import json
from datetime import datetime

from .config import RedactionConfig, RedactionMode, DetectionResult, ProcessingMetadata
from .logger import LoggerMixin
from .synthetic_data import SyntheticDataGenerator


class Redactor(LoggerMixin):
    """
    Main redaction class for applying blur or mask effects to detected PII regions.
    """
    
    def __init__(self, config: RedactionConfig):
        """
        Initialize redactor.
        
        Args:
            config: Redaction configuration
        """
        self.config = config
        
        # Initialize synthetic data generator for text replacement
        self.synthetic_generator = SyntheticDataGenerator(seed=config.synthetic_seed)
    
    def redact_image(
        self,
        image: Union[np.ndarray, str, Path],
        detections: List[DetectionResult],
        mode: Optional[RedactionMode] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply redaction to image based on detections.
        
        Args:
            image: Input image as numpy array, file path, or Path object
            detections: List of PII detections to redact
            mode: Redaction mode (blur or mask)
            output_path: Optional path to save redacted image
            
        Returns:
            Tuple of (redacted_image_array, metadata_dict)
        """
        # Load image if needed
        image_array = self._load_image(image)
        original_shape = image_array.shape
        
        # Use default mode if not specified
        if mode is None:
            mode = self.config.default_mode
        
        # Apply redaction
        redacted_image = image_array.copy()
        
        for detection in detections:
            redacted_image = self._apply_single_redaction(
                redacted_image, detection, mode
            )
        
        # Save image if output path provided
        if output_path:
            self._save_image(redacted_image, output_path)
        
        # Create metadata
        metadata = {
            "redaction_mode": mode.value,
            "total_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
            "original_shape": original_shape,
            "redacted_regions": len(detections),
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_info(f"Applied {mode.value} redaction to {len(detections)} regions")
        return redacted_image, metadata
    
    def _load_image(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load image as numpy array."""
        if isinstance(image, np.ndarray):
            return image
        
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load with PIL for better format support
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to BGR for OpenCV compatibility
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.log_error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _save_image(self, image_array: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save image array to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert BGR to RGB for PIL
            if len(image_array.shape) == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image_array)
            
            # Save with appropriate format
            pil_image.save(output_path)
            self.log_info(f"Saved redacted image to {output_path}")
            
        except Exception as e:
            self.log_error(f"Failed to save image to {output_path}: {e}")
            raise
    
    def _apply_single_redaction(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        mode: RedactionMode
    ) -> np.ndarray:
        """
        Apply redaction to a single detected region.
        
        Args:
            image: Image array to modify
            detection: Detection result with bounding box
            mode: Redaction mode
            
        Returns:
            Modified image array
        """
        x1, y1, x2, y2 = detection.bbox
        
        # Add padding
        padding = self.config.padding_pixels
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Ensure valid coordinates
        if x2 <= x1 or y2 <= y1:
            self.log_warning(f"Invalid bounding box for {detection.pii_type}: {detection.bbox}")
            return image
        
        try:
            if mode == RedactionMode.BLUR:
                return self._apply_blur(image, (x1, y1, x2, y2))
            elif mode == RedactionMode.MASK:
                return self._apply_mask(image, (x1, y1, x2, y2))
            elif mode == RedactionMode.SYNTHETIC:
                return self._apply_synthetic_replacement(image, detection, (x1, y1, x2, y2))
            else:
                self.log_warning(f"Unknown redaction mode: {mode}")
                return image
                
        except Exception as e:
            self.log_error(f"Failed to apply {mode.value} redaction: {e}")
            return image
    
    def _apply_blur(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply Gaussian blur to specified region.
        
        Args:
            image: Image array
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Image with blurred region
        """
        x1, y1, x2, y2 = bbox
        
        # Extract region
        region = image[y1:y2, x1:x2].copy()
        
        if region.size == 0:
            return image
        
        # Apply Gaussian blur
        kernel_size = self.config.blur_kernel_size
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred_region = cv2.GaussianBlur(
            region,
            (kernel_size, kernel_size),
            self.config.blur_sigma
        )
        
        # Replace region in original image
        result_image = image.copy()
        result_image[y1:y2, x1:x2] = blurred_region
        
        return result_image
    
    def _apply_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply solid color mask to specified region.
        
        Args:
            image: Image array
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Image with masked region
        """
        x1, y1, x2, y2 = bbox
        
        # Create mask color (BGR format for OpenCV)
        mask_color_bgr = (
            self.config.mask_color[2],  # B
            self.config.mask_color[1],  # G
            self.config.mask_color[0]   # R
        )
        
        # Apply mask
        result_image = image.copy()
        result_image[y1:y2, x1:x2] = mask_color_bgr
        
        return result_image
    
    def _apply_synthetic_replacement(
        self, 
        image: np.ndarray, 
        detection: DetectionResult, 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Replace detected PII text with synthetic data using advanced text rendering.
        
        Args:
            image: Image array
            detection: Detection result containing PII type and text
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Image with synthetic text replacement
        """
        x1, y1, x2, y2 = bbox
        
        try:
            # Generate synthetic replacement text - ENSURE IT'S ACTUALLY DIFFERENT
            original_text = detection.text if hasattr(detection, 'text') and detection.text else ""
            pii_type = detection.pii_type.value if hasattr(detection.pii_type, 'value') else str(detection.pii_type)
            
            # Generate replacement multiple times if needed to ensure it's different
            synthetic_text = self._generate_different_replacement(pii_type, original_text)
            
            self.log_info(f"🔄 Replacing '{original_text}' → '{synthetic_text}' ({pii_type})")
            
            # ENHANCED text replacement with GREAT quality
            result_image = self._render_replacement_text(
                image, bbox, original_text, synthetic_text
            )
            
            return result_image
            
        except Exception as e:
            self.log_error(f"Failed to apply synthetic replacement: {e}")
            import traceback
            self.log_error(f"Full error: {traceback.format_exc()}")
            # Fallback to blur if synthetic replacement fails
            return self._apply_blur(image, bbox)
    
    def _generate_different_replacement(self, pii_type: str, original_text: str) -> str:
        """Generate replacement text that is guaranteed to be different from original."""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                synthetic_text = self.synthetic_generator.generate_replacement(
                    pii_type=pii_type,
                    original_text=original_text,
                    prefer_indian=self.config.prefer_indian_names
                )
                
                # Ensure the replacement is actually different
                if synthetic_text.lower().strip() != original_text.lower().strip():
                    return synthetic_text
                
                # If same, try again with different approach
                self.log_warning(f"Generated same text '{synthetic_text}', retrying...")
                
            except Exception as e:
                self.log_error(f"Error in replacement attempt {attempt + 1}: {e}")
        
        # Fallback: force different replacement with PROPER synthetic data
        self.log_warning(f"All attempts failed, using fallback replacement for '{pii_type}'")
        
        if pii_type in ['phone', 'phone_number']:
            return "8765432109" if original_text != "8765432109" else "7654321098"
        elif pii_type in ['email', 'email_address']:
            return "john.doe@example.com" if original_text != "john.doe@example.com" else "jane.smith@test.org"
        elif pii_type in ['person_name', 'name']:
            return "Raj Kumar" if original_text != "Raj Kumar" else "Priya Sharma"
        elif pii_type in ['date', 'birth_date', 'dob']:
            return "15/05/1990" if original_text != "15/05/1990" else "22/08/1985"
        else:
            # Generate proper random replacement instead of "REPLACED_"
            import string
            import random
            if original_text.isdigit():
                # For numeric data, generate random numbers
                return ''.join(random.choices(string.digits, k=len(original_text)))
            elif original_text.isalpha():
                # For alphabetic data, generate random letters
                return ''.join(random.choices(string.ascii_letters, k=len(original_text)))
            else:
                # For mixed data, generate random alphanumeric
                return ''.join(random.choices(string.ascii_letters + string.digits, k=len(original_text)))
    
    def _render_replacement_text(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        original_text: str, 
        synthetic_text: str
    ) -> np.ndarray:
        """ENHANCED text rendering with proper sizing for GREAT real-world quality."""
        x1, y1, x2, y2 = bbox
        region_width = x2 - x1
        region_height = y2 - y1
        
        if region_width <= 0 or region_height <= 0:
            self.log_warning(f"Invalid region dimensions: {region_width}x{region_height}")
            return image
        
        result_image = image.copy()
        
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            
            # Extract background color for seamless blending
            bg_color = self._extract_background_color(image, bbox)
            
            # Calculate optimal scale factor based on region size
            # Larger regions get higher resolution for better quality
            if region_width * region_height > 10000:  # Large regions
                scale_factor = 4
            elif region_width * region_height > 5000:   # Medium regions
                scale_factor = 3  
            else:                                       # Small regions
                scale_factor = 2
            
            canvas_width = region_width * scale_factor
            canvas_height = region_height * scale_factor
            
            # Create high-resolution canvas
            canvas = PILImage.new('RGB', (canvas_width, canvas_height), color=bg_color)
            draw = ImageDraw.Draw(canvas)
            
            # ENHANCED font size calculation
            base_font_size = self._calculate_optimal_font_size(
                original_text, synthetic_text, region_width, region_height, scale_factor
            )
            
            # Find the best available font
            optimal_font = self._get_best_font(base_font_size)
            
            # Fine-tune size to ensure perfect fit
            final_font, final_size = self._fine_tune_font_size(
                draw, optimal_font, synthetic_text, canvas_width, canvas_height
            )
            
            self.log_debug(f"RENDER: '{synthetic_text}' with font size {final_size//scale_factor} in {region_width}x{region_height}")
            
            # Enhanced text positioning with better centering
            text_bbox = draw.textbbox((0, 0), synthetic_text, font=final_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Better centering with slight upward bias for readability
            text_x = max(0, (canvas_width - text_width) // 2)
            text_y = max(0, (canvas_height - text_height) // 2 - (text_height // 10))
            
            # Draw text with high quality
            draw.text((text_x, text_y), synthetic_text, fill='black', font=final_font)
            
            # High-quality downscaling with anti-aliasing
            final_region = canvas.resize((region_width, region_height), PILImage.LANCZOS)
            
            # Apply Gaussian blur for smoother edges (very subtle)
            from PIL import ImageFilter
            final_region = final_region.filter(ImageFilter.GaussianBlur(radius=0.3))
            
            # Convert and place in image
            region_array = np.array(final_region)
            result_image[y1:y2, x1:x2] = cv2.cvtColor(region_array, cv2.COLOR_RGB2BGR)
            
            return result_image
            
        except Exception as e:
            self.log_error(f"Enhanced text rendering failed: {e}")
            import traceback
            self.log_error(f"Rendering error details: {traceback.format_exc()}")
            
            # Improved fallback rendering
            try:
                # Clear region with background color
                result_image[y1:y2, x1:x2] = bg_color[::-1]  # Convert RGB to BGR
                
                # Try simple text rendering
                cv2.putText(
                    result_image, 
                    synthetic_text, 
                    (x1 + 2, y1 + region_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    min(1.0, (region_height / 20)), 
                    (0, 0, 0), 
                    1
                )
                
            except Exception as fallback_error:
                self.log_error(f"Fallback rendering also failed: {fallback_error}")
                # Last resort - just clear the region
                result_image[y1:y2, x1:x2] = (255, 255, 255)
            
            return result_image

    def _calculate_optimal_font_size(
        self, 
        original_text: str, 
        synthetic_text: str, 
        region_width: int, 
        region_height: int, 
        scale_factor: int
    ) -> int:
        """Calculate optimal font size based on region dimensions and text length."""
        # Base size calculation considering both dimensions
        height_based = max(12, region_height - 4) * scale_factor
        width_based = max(10, (region_width // max(1, len(synthetic_text))) * 1.5) * scale_factor
        
        # Use smaller to ensure fit, but ensure minimum readability
        base_size = min(height_based, width_based)
        base_size = max(base_size, 14 * scale_factor)  # Minimum readable size
        
        # Adjust based on text length difference
        length_ratio = len(synthetic_text) / max(1, len(original_text))
        if length_ratio > 1.2:  # New text is longer
            base_size = int(base_size * 0.9)
        elif length_ratio < 0.8:  # New text is shorter
            base_size = int(base_size * 1.1)
        
        return base_size
    
    def _get_best_font(self, base_size: int):
        """Get the best available font for rendering."""
        from PIL import ImageFont
        
        # Priority list of fonts for best readability
        font_candidates = [
            "arial.ttf", "Arial.ttf",           # Clear, widely available
            "calibri.ttf", "Calibri.ttf",       # Modern, readable
            "segoeui.ttf", "SegoeUI.ttf",       # Windows 10/11 default
            "helvetica.ttf", "Helvetica.ttf",   # Classic sans-serif
            "verdana.ttf", "Verdana.ttf",       # Good for small sizes
            "tahoma.ttf", "Tahoma.ttf",         # Compact but readable
        ]
        
        for font_name in font_candidates:
            try:
                font = ImageFont.truetype(font_name, base_size)
                # Test that font works by getting a text bbox
                return font
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        return ImageFont.load_default()
    
    def _fine_tune_font_size(
        self, 
        draw, 
        base_font, 
        text: str, 
        canvas_width: int, 
        canvas_height: int
    ):
        """Fine-tune font size to ensure perfect fit within canvas."""
        from PIL import ImageFont
        
        # Extract base size if possible
        try:
            base_size = base_font.size if hasattr(base_font, 'size') else 24
            font_path = base_font.path if hasattr(base_font, 'path') else None
        except:
            base_size = 24
            font_path = None
        
        # Target utilization (don't use full canvas to avoid cramped text)
        target_width = canvas_width * 0.85
        target_height = canvas_height * 0.75
        
        # Binary search for optimal size
        min_size = max(8, base_size // 2)
        max_size = base_size * 2
        optimal_size = base_size
        best_font = base_font
        
        for _ in range(10):  # Limit iterations
            test_size = (min_size + max_size) // 2
            
            try:
                if font_path:
                    test_font = ImageFont.truetype(font_path, test_size)
                else:
                    test_font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=test_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                fits_width = text_width <= target_width
                fits_height = text_height <= target_height
                
                if fits_width and fits_height:
                    # Size works, try larger
                    optimal_size = test_size
                    best_font = test_font
                    min_size = test_size + 1
                else:
                    # Too big, try smaller
                    max_size = test_size - 1
                    
            except:
                max_size = test_size - 1
                
            if min_size > max_size:
                break
        
        return best_font, optimal_size
    
    def _extract_background_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> tuple:
        """Extract the most common background color around the text region."""
        x1, y1, x2, y2 = bbox
        
        try:
            # Sample area around the bbox
            margin = 5
            sample_x1 = max(0, x1 - margin)
            sample_y1 = max(0, y1 - margin)
            sample_x2 = min(image.shape[1], x2 + margin)
            sample_y2 = min(image.shape[0], y2 + margin)
            
            # Extract sample region
            sample_region = image[sample_y1:sample_y2, sample_x1:sample_x2]
            
            # Convert BGR to RGB
            sample_rgb = cv2.cvtColor(sample_region, cv2.COLOR_BGR2RGB)
            
            # Get the most common color (mode)
            pixels = sample_rgb.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            most_common_color = unique_colors[np.argmax(counts)]
            
            return tuple(most_common_color)
            
        except:
            return (255, 255, 255)  # Default to white
    
    def _find_best_font_size(
        self, 
        draw, 
        text: str, 
        max_width: int, 
        max_height: int, 
        scale_factor: int = 1
    ) -> Tuple[any, int]:
        """Find the best font size that fits the given dimensions."""
        from PIL import ImageFont
        
        # System font paths with more options
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/times.ttf",
            "arial.ttf", "Arial.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/arial.ttf"
        ]
        
        # Calculate target size based on region dimensions
        base_size = min(max_height // 2, max_width // len(text)) * scale_factor
        base_size = max(base_size, 20 * scale_factor)  # Minimum readable size
        
        self.log_debug(f"Starting font search with base size: {base_size} for region {max_width}x{max_height}")
        
        best_font = None
        best_size = 0
        
        # Try sizes from large to small
        for size in range(base_size, 10 * scale_factor, -2 * scale_factor):
            font = None
            
            # Try to load a system font
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, size)
                    break
                except:
                    continue
            
            # Fallback to default font
            if font is None:
                try:
                    font = ImageFont.load_default()
                except:
                    continue
            
            # Test if text fits
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Check if it fits with some margin
                if (text_width <= max_width * 0.95 and 
                    text_height <= max_height * 0.95):
                    best_font = font
                    best_size = size
                    break
                    
            except:
                continue
        
        # If no font found, use minimum size
        if best_font is None:
            for font_path in font_paths:
                try:
                    best_font = ImageFont.truetype(font_path, 15 * scale_factor)
                    best_size = 15 * scale_factor
                    break
                except:
                    continue
            
            if best_font is None:
                best_font = ImageFont.load_default()
                best_size = 10
        
        self.log_debug(f"Selected font size: {best_size // scale_factor} (scaled: {best_size})")
        return best_font, best_size
    
    def _get_optimal_font(self, draw, text: str, max_width: int, max_height: int):
        """Find optimal font size that fits within the given dimensions."""
        from PIL import ImageFont
        
        # Start with a reasonable base size
        base_size = max(int(max_height * 0.7), 14)  # Increased minimum size
        font_paths = [
            "arial.ttf", "Arial.ttf",  # Windows
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/TTF/arial.ttf"  # Some Linux distros
        ]
        
        # Try different font sizes, starting from base_size and going down if needed
        for size in range(base_size, 8, -1):  # Don't go below 8px
            font = None
            
            # Try to load system fonts
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, size)
                    break
                except (OSError, IOError):
                    continue
            
            # Fallback to default font if no system fonts work
            if font is None:
                try:
                    font = ImageFont.load_default()
                except:
                    # Ultimate fallback - create a dummy font-like object
                    class DummyFont:
                        def getsize(self, text):
                            return (len(text) * 8, 12)
                    font = DummyFont()
            
            # Check if text fits within dimensions
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Add some padding (10% margin)
                if text_width <= max_width * 0.9 and text_height <= max_height * 0.9:
                    return font
            except:
                # If textbbox fails, use approximate sizing
                if hasattr(font, 'getsize'):
                    text_width, text_height = font.getsize(text)
                else:
                    text_width, text_height = len(text) * size * 0.6, size
                
                if text_width <= max_width * 0.9 and text_height <= max_height * 0.9:
                    return font
        
        # If no size fits, return smallest font
        try:
            return ImageFont.truetype(font_paths[0], 8)
        except:
            return ImageFont.load_default()
    
    def reset_synthetic_session(self):
        """Reset synthetic data generator for new document session."""
        self.synthetic_generator.reset_session()
        self.log_debug("Reset synthetic data generator session")
    
    def redact_multiple_images(
        self,
        input_paths: List[Union[str, Path]],
        detection_results: List[List[DetectionResult]],
        output_dir: Union[str, Path],
        mode: Optional[RedactionMode] = None
    ) -> List[ProcessingMetadata]:
        """
        Redact multiple images in batch.
        
        Args:
            input_paths: List of input image paths
            detection_results: List of detection results for each image
            output_dir: Output directory for redacted images
            mode: Redaction mode
            
        Returns:
            List of processing metadata for each image
        """
        if len(input_paths) != len(detection_results):
            raise ValueError("Number of input paths must match number of detection results")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        for input_path, detections in zip(input_paths, detection_results):
            input_path = Path(input_path)
            output_path = output_dir / f"redacted_{input_path.name}"
            
            try:
                start_time = datetime.now()
                
                # Apply redaction
                redacted_image, redaction_metadata = self.redact_image(
                    input_path, detections, mode, output_path
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds() * 1000
                
                # Create processing metadata
                metadata = ProcessingMetadata(
                    input_file=str(input_path),
                    output_file=str(output_path),
                    processing_time_ms=processing_time,
                    detections=detections,
                    ocr_engine_used="unknown",  # This should be set by the caller
                    redaction_mode=mode or self.config.default_mode,
                    timestamp=start_time.isoformat()
                )
                
                metadata_list.append(metadata)
                
                # Save metadata JSON
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
            except Exception as e:
                self.log_error(f"Failed to process {input_path}: {e}")
        
        self.log_info(f"Batch redaction complete: {len(metadata_list)} images processed")
        return metadata_list
    
    def create_redaction_mask(
        self,
        image_shape: Tuple[int, int],
        detections: List[DetectionResult]
    ) -> np.ndarray:
        """
        Create a binary mask showing redacted regions.
        
        Args:
            image_shape: Shape of the original image (height, width)
            detections: List of detections
            
        Returns:
            Binary mask array (255 for redacted regions, 0 elsewhere)
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Add padding
            padding = self.config.padding_pixels
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            # Fill mask region
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def get_redaction_statistics(self, detections: List[DetectionResult]) -> Dict[str, Any]:
        """
        Calculate statistics about redactions.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with redaction statistics
        """
        if not detections:
            return {
                "total_detections": 0,
                "pii_type_counts": {},
                "confidence_stats": {},
                "total_area_pixels": 0
            }
        
        # Count by PII type
        pii_counts = {}
        confidences = []
        total_area = 0
        
        for detection in detections:
            # Count by type
            pii_type = detection.pii_type.value
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
            
            # Collect confidence scores
            confidences.append(detection.confidence)
            
            # Calculate area
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            total_area += area
        
        # Calculate confidence statistics
        confidence_stats = {
            "mean": np.mean(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "std": np.std(confidences)
        }
        
        return {
            "total_detections": len(detections),
            "pii_type_counts": pii_counts,
            "confidence_stats": confidence_stats,
            "total_area_pixels": total_area
        }

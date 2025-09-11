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

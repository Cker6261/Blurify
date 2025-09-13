"""
PII detection using spaCy NER, regex patterns, and optional Presidio integration.

Detects emails, phone numbers, names, dates, and Indian government IDs.
"""

import re
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass

from .config import DetectionConfig, PIIType, DetectionResult
from .ocr import OCRResult
from .logger import LoggerMixin


@dataclass
class RegexPattern:
    """Regex pattern for PII detection."""
    pattern: re.Pattern
    pii_type: PIIType
    confidence: float = 0.9
    description: str = ""


class RegexDetector(LoggerMixin):
    """Regex-based PII detector."""
    
    def __init__(self):
        """Initialize regex patterns."""
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[RegexPattern]:
        """Compile all regex patterns."""
        patterns = []
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        patterns.append(RegexPattern(
            pattern=re.compile(email_pattern, re.IGNORECASE),
            pii_type=PIIType.EMAIL,
            confidence=0.95,
            description="Standard email format"
        ))
        
        # Indian phone numbers (10 digits, with optional +91, spaces, hyphens)
        indian_phone_patterns = [
            r'\+91[-\s]?[6-9]\d{4}[-\s]?\d{5}',  # +91 format
            r'\b[6-9]\d{4}[-\s]?\d{5}\b',        # 10 digit format
            r'\b0[1-9]\d{2}[-\s]?\d{3}[-\s]?\d{4}\b'  # Landline format
        ]
        
        for i, pattern in enumerate(indian_phone_patterns):
            patterns.append(RegexPattern(
                pattern=re.compile(pattern),
                pii_type=PIIType.PHONE,
                confidence=0.9 - i * 0.1,  # Decreasing confidence
                description=f"Indian phone pattern {i+1}"
            ))
        
        # International phone (basic pattern)
        patterns.append(RegexPattern(
            pattern=re.compile(r'\+\d{1,3}[-\s]?\d{1,14}'),
            pii_type=PIIType.PHONE,
            confidence=0.8,
            description="International phone"
        ))
        
        # Aadhaar-like 12-digit numbers (with optional spaces/hyphens)
        aadhaar_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        patterns.append(RegexPattern(
            pattern=re.compile(aadhaar_pattern),
            pii_type=PIIType.AADHAAR,
            confidence=0.85,
            description="12-digit Aadhaar-like number"
        ))
        
        # PAN card pattern (5 letters, 4 digits, 1 letter)
        pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
        patterns.append(RegexPattern(
            pattern=re.compile(pan_pattern),
            pii_type=PIIType.PAN,
            confidence=0.95,
            description="PAN card format"
        ))
        
        # Date patterns (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',        # DD/MM/YYYY or MM/DD/YYYY
            r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',        # YYYY/MM/DD
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # 15 Jan 2023
        ]
        
        for i, pattern in enumerate(date_patterns):
            patterns.append(RegexPattern(
                pattern=re.compile(pattern, re.IGNORECASE),
                pii_type=PIIType.DATE,
                confidence=0.8 - i * 0.05,
                description=f"Date pattern {i+1}"
            ))
        
        self.log_info(f"Compiled {len(patterns)} regex patterns")
        return patterns
    
    def detect(self, text: str) -> List[Tuple[PIIType, str, float, Tuple[int, int]]]:
        """
        Detect PII in text using regex patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (pii_type, matched_text, confidence, (start, end)) tuples
        """
        detections = []
        
        for pattern_info in self.patterns:
            for match in pattern_info.pattern.finditer(text):
                matched_text = match.group().strip()
                if len(matched_text) >= 3:  # Filter out very short matches
                    detections.append((
                        pattern_info.pii_type,
                        matched_text,
                        pattern_info.confidence,
                        (match.start(), match.end())
                    ))
        
        return detections


class SpacyNERDetector(LoggerMixin):
    """spaCy-based Named Entity Recognition detector."""
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize spaCy NER detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.nlp = None
        self._initialize_spacy()
    
    def _initialize_spacy(self) -> None:
        """Initialize spaCy model with safe memory handling."""
        # Set spaCy to None by default to prevent crashes
        self.nlp = None
        
        try:
            import spacy
            import gc
            import os
            
            # Try to check available memory if psutil is available
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                if available_memory_gb < 1.0:  # Need at least 1GB free
                    self.log_warning(f"Insufficient memory ({available_memory_gb:.1f}GB available). spaCy disabled to prevent system instability.")
                    return
                self.log_info(f"Memory check: {available_memory_gb:.1f}GB available")
            except ImportError:
                self.log_info("psutil not available, skipping memory check")
            
            # Force garbage collection before loading memory-intensive model
            gc.collect()
            
            # Check if user explicitly disabled spaCy
            if os.environ.get('BLURIFY_DISABLE_SPACY', '').lower() == 'true':
                self.log_info("spaCy disabled via environment variable BLURIFY_DISABLE_SPACY")
                return
            
            self.log_info("Attempting to load spaCy model...")
            
            # Try to load with very minimal components and catch ALL exceptions
            try:
                # Load with absolute minimal components to reduce memory usage
                self.nlp = spacy.load(
                    self.config.spacy_model, 
                    exclude=["parser", "tagger", "lemmatizer", "textcat", "custom"]
                )
                self.log_info(f"✅ Loaded spaCy model: {self.config.spacy_model} (minimal)")
            except Exception as e:
                self.log_warning(f"Failed to load {self.config.spacy_model}: {str(e)}")
                try:
                    # Fallback to en_core_web_sm with minimal components
                    self.nlp = spacy.load(
                        "en_core_web_sm", 
                        exclude=["parser", "tagger", "lemmatizer", "textcat", "custom"]
                    )
                    self.log_info("✅ Loaded fallback: en_core_web_sm (minimal)")
                except Exception as fallback_error:
                    self.log_warning(f"spaCy fallback failed: {str(fallback_error)}")
                    self.log_info("📝 Name detection disabled - using regex-only mode for safety")
                    self.nlp = None
                    
        except ImportError as e:
            self.log_warning(f"spaCy/psutil not available: {str(e)}")
            self.nlp = None
        except Exception as e:
            self.log_warning(f"Unexpected error initializing spaCy: {str(e)}")
            self.nlp = None
    
    def detect(self, text: str) -> List[Tuple[PIIType, str, float, Tuple[int, int]]]:
        """
        Detect PII using spaCy NER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (pii_type, matched_text, confidence, (start, end)) tuples
        """
        if self.nlp is None:
            self.log_warning("spaCy not available, skipping NER detection")
            return []
        
        detections = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                pii_type = None
                confidence = 0.8  # Base confidence for spaCy NER
                
                # Map spaCy entity types to our PII types
                if ent.label_ == "PERSON":
                    pii_type = PIIType.PERSON_NAME
                    confidence = 0.85
                elif ent.label_ == "DATE":
                    pii_type = PIIType.DATE
                    confidence = 0.8
                # Add more mappings as needed
                
                if pii_type and confidence >= self.config.confidence_threshold:
                    detections.append((
                        pii_type,
                        ent.text.strip(),
                        confidence,
                        (ent.start_char, ent.end_char)
                    ))
        
        except Exception as e:
            self.log_error(f"spaCy NER detection failed: {e}")
        
        return detections


class PresidioDetector(LoggerMixin):
    """Microsoft Presidio-based PII detector (optional)."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize Presidio detector."""
        self.config = config
        self.analyzer = None
        
        if config.use_presidio:
            self._initialize_presidio()
    
    def _initialize_presidio(self) -> None:
        """Initialize Presidio analyzer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            
            # TODO: Add Presidio configuration
            # This is a placeholder implementation
            # Users need to install: pip install presidio-analyzer presidio-anonymizer
            
            provider = NlpEngineProvider()
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            nlp_engine = provider.create_engine(nlp_configuration)
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            
            self.log_info("Presidio analyzer initialized")
            
        except ImportError:
            self.log_warning(
                "Presidio not available. Install with: "
                "pip install presidio-analyzer presidio-anonymizer"
            )
            self.analyzer = None
        except Exception as e:
            self.log_error(f"Failed to initialize Presidio: {e}")
            self.analyzer = None
    
    def detect(self, text: str) -> List[Tuple[PIIType, str, float, Tuple[int, int]]]:
        """
        Detect PII using Presidio.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (pii_type, matched_text, confidence, (start, end)) tuples
        """
        if self.analyzer is None:
            return []
        
        detections = []
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language='en',
                score_threshold=self.config.presidio_threshold
            )
            
            for result in results:
                # Map Presidio entity types to our PII types
                pii_type = self._map_presidio_entity(result.entity_type)
                if pii_type:
                    matched_text = text[result.start:result.end]
                    detections.append((
                        pii_type,
                        matched_text,
                        result.score,
                        (result.start, result.end)
                    ))
        
        except Exception as e:
            self.log_error(f"Presidio detection failed: {e}")
        
        return detections
    
    def _map_presidio_entity(self, entity_type: str) -> Optional[PIIType]:
        """Map Presidio entity types to our PII types."""
        mapping = {
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "PERSON": PIIType.PERSON_NAME,
            "DATE_TIME": PIIType.DATE,
            # Add more mappings as needed
        }
        return mapping.get(entity_type)


class PIIDetector(LoggerMixin):
    """
    Main PII detector that combines multiple detection methods.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize PII detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        
        # Initialize sub-detectors with error handling
        try:
            self.regex_detector = RegexDetector()
            self.log_info("✅ Regex detector initialized")
        except Exception as e:
            self.log_error(f"Failed to initialize regex detector: {e}")
            raise
        
        if config.use_spacy:
            try:
                self.spacy_detector = SpacyNERDetector(config)
                if self.spacy_detector.nlp is None:
                    self.log_warning("⚠️ spaCy detector initialized but model failed to load - name detection disabled")
                else:
                    self.log_info("✅ spaCy detector initialized")
            except Exception as e:
                self.log_warning(f"⚠️ spaCy detector failed to initialize: {e}")
                # Create a dummy detector that won't crash the system
                self.spacy_detector = None
        else:
            self.log_info("spaCy detector disabled by configuration")
            self.spacy_detector = None
        
        try:
            self.presidio_detector = PresidioDetector(config) if config.use_presidio else None
            if self.presidio_detector:
                self.log_info("✅ Presidio detector initialized")
        except Exception as e:
            self.log_warning(f"⚠️ Presidio detector failed to initialize: {e}")
            self.presidio_detector = None
        
        self.log_info("PII detector initialization completed")
    
    def detect_in_text(self, text: str) -> List[DetectionResult]:
        """
        Detect PII in plain text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detection results
        """
        all_detections = []
        
        # Regex detection
        regex_results = self.regex_detector.detect(text)
        for pii_type, matched_text, confidence, (start, end) in regex_results:
            if pii_type in self.config.enabled_pii_types:
                # For text-only detection, we don't have bbox coordinates
                all_detections.append(DetectionResult(
                    pii_type=pii_type,
                    text=matched_text,
                    bbox=(0, 0, 0, 0),  # Placeholder bbox
                    confidence=confidence,
                    source="regex"
                ))
        
        # spaCy NER detection (if available)
        if self.spacy_detector and self.spacy_detector.nlp is not None:
            spacy_results = self.spacy_detector.detect(text)
            for pii_type, matched_text, confidence, (start, end) in spacy_results:
                if pii_type in self.config.enabled_pii_types:
                    all_detections.append(DetectionResult(
                        pii_type=pii_type,
                        text=matched_text,
                        bbox=(0, 0, 0, 0),  # Placeholder bbox
                        confidence=confidence,
                        source="spacy"
                    ))
        
        # Presidio detection (if enabled)
        if self.presidio_detector:
            presidio_results = self.presidio_detector.detect(text)
            for pii_type, matched_text, confidence, (start, end) in presidio_results:
                if pii_type in self.config.enabled_pii_types:
                    all_detections.append(DetectionResult(
                        pii_type=pii_type,
                        text=matched_text,
                        bbox=(0, 0, 0, 0),  # Placeholder bbox
                        confidence=confidence,
                        source="presidio"
                    ))
        
        # Remove duplicates and filter by confidence
        filtered_detections = self._deduplicate_detections(all_detections)
        
        self.log_info(f"Detected {len(filtered_detections)} PII items in text")
        return filtered_detections
    
    def detect_in_ocr_results(self, ocr_results: List[OCRResult]) -> List[DetectionResult]:
        """
        Detect PII in OCR results, mapping text detections back to bounding boxes.
        
        Args:
            ocr_results: List of OCR results with text and bounding boxes
            
        Returns:
            List of detection results with proper bounding boxes
        """
        all_detections = []
        
        # Process each OCR result separately to preserve bounding boxes
        for ocr_result in ocr_results:
            text_detections = self.detect_in_text(ocr_result.text)
            
            for detection in text_detections:
                # Use the OCR bounding box for this detection
                detection.bbox = ocr_result.bbox
                all_detections.append(detection)
        
        # Also try to detect PII in the full combined text for better context
        full_text = " ".join([ocr.text for ocr in ocr_results])
        if full_text.strip():
            full_text_detections = self.detect_in_text(full_text)
            
            # Try to map full-text detections back to OCR bounding boxes
            for detection in full_text_detections:
                mapped_bbox = self._map_text_to_bbox(detection.text, ocr_results)
                if mapped_bbox:
                    detection.bbox = mapped_bbox
                    all_detections.append(detection)
        
        # Remove duplicates and filter
        filtered_detections = self._deduplicate_detections(all_detections)
        
        self.log_info(f"Detected {len(filtered_detections)} PII items in OCR results")
        return filtered_detections
    
    def _map_text_to_bbox(self, search_text: str, ocr_results: List[OCRResult]) -> Optional[Tuple[int, int, int, int]]:
        """
        Map detected text back to OCR bounding box.
        
        Args:
            search_text: Text to find in OCR results
            ocr_results: List of OCR results
            
        Returns:
            Bounding box if found, None otherwise
        """
        search_text_lower = search_text.lower().strip()
        
        # First, try exact match
        for ocr_result in ocr_results:
            if search_text_lower in ocr_result.text.lower():
                return ocr_result.bbox
        
        # Try partial matches
        for ocr_result in ocr_results:
            if any(word in ocr_result.text.lower() for word in search_text_lower.split() if len(word) > 2):
                return ocr_result.bbox
        
        return None
    
    def _deduplicate_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Remove duplicate detections and filter by confidence.
        
        Args:
            detections: List of detection results
            
        Returns:
            Filtered list of unique detections
        """
        # Filter by confidence threshold
        filtered = [d for d in detections if d.confidence >= self.config.confidence_threshold]
        
        # Group by text and type, keep highest confidence
        seen: Dict[Tuple[PIIType, str], DetectionResult] = {}
        
        for detection in filtered:
            key = (detection.pii_type, detection.text.lower().strip())
            if key not in seen or detection.confidence > seen[key].confidence:
                seen[key] = detection
        
        return list(seen.values())

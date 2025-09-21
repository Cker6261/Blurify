"""
Context-aware PII detection filters to reduce false positives.

Filters out common labels, headings, and non-PII text that might be mistakenly 
detected as PII by regex or NLP models.
"""

import re
from typing import List, Set, Tuple
from enum import Enum

from .config import PIIType, DetectionResult
from .logger import LoggerMixin


class ContextType(Enum):
    """Types of text context."""
    LABEL = "label"          # Form labels like "Name:", "Email:"
    HEADING = "heading"      # Document headings  
    INSTRUCTION = "instruction"  # Instructions like "Enter your name"
    ACTUAL_PII = "actual_pii"   # Actual PII data


class PIIContextFilter(LoggerMixin):
    """Filters PII detections based on context to reduce false positives."""
    
    def __init__(self):
        """Initialize context filter with common patterns."""
        
        # Common form labels that are NOT PII
        self.common_labels = {
            PIIType.EMAIL: {
                'email', 'email address', 'e-mail', 'email id', 'email:', 'e-mail:', 
                'email address:', 'electronic mail', 'mail', 'mail id', 'mail:'
            },
            PIIType.PHONE: {
                'phone', 'phone number', 'phone no', 'phone:', 'phone number:', 
                'mobile', 'mobile number', 'mobile no', 'mobile:', 'contact', 
                'contact number', 'tel', 'telephone', 'cell', 'cell number'
            },
            PIIType.PERSON_NAME: {
                'name', 'name:', 'full name', 'full name:', 'first name', 'last name',
                'surname', 'given name', 'applicant name', 'candidate name', 
                'student name', 'employee name', 'customer name', 'user name',
                'name of', 'names', 'signatory', 'signature', 'signed by'
            },
            PIIType.DATE: {  
                'date', 'date:', 'birth date', 'date of birth', 'dob', 'dob:',
                'joining date', 'issue date', 'valid till', 'expiry date',
                'application date', 'submission date', 'created on', 'updated on'
            },
            PIIType.AADHAAR: {
                'aadhaar', 'aadhaar number', 'aadhaar no', 'aadhaar:', 'uid',
                'unique id', 'aadhaar card', 'aadhaar number:'
            },
            PIIType.PAN: {
                'pan', 'pan number', 'pan no', 'pan:', 'pan card', 'pan number:',
                'permanent account number', 'income tax pan'
            }
        }
        
        # Instructions that typically precede actual PII
        self.instruction_patterns = {
            r'enter\s+your\s+\w+',
            r'provide\s+your\s+\w+', 
            r'fill\s+in\s+your\s+\w+',
            r'write\s+your\s+\w+',
            r'mention\s+your\s+\w+',
            r'input\s+your\s+\w+',
            r'type\s+your\s+\w+'
        }
        
        # Compile instruction patterns
        self.instruction_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.instruction_patterns]
        
        # Patterns that indicate actual PII (not labels)
        self.actual_pii_indicators = {
            PIIType.EMAIL: [
                r'@\w+\.',  # Contains @ and domain
                r'\w+@\w+\.\w+',  # Full email pattern
            ],
            PIIType.PHONE: [
                r'\+\d{1,3}[\s-]?\d',  # International prefix
                r'\d{10,}',  # Long number sequences
                r'\d{3,4}[\s-]\d{3,4}[\s-]\d{4}',  # Formatted phone patterns
            ],
            PIIType.PERSON_NAME: [
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # Proper case first/last name
                r'^[A-Z]\.\s*[A-Z][a-z]+',  # Initial + last name
                r'^[A-Z][a-z]+\s+[A-Z]\.',  # First name + initial
            ],
            PIIType.DATE: [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date formats
                r'\d{1,2}\s+\w+\s+\d{4}',  # DD Month YYYY
                r'\w+\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
            ]
        }

    def classify_context(self, text: str, pii_type: PIIType) -> ContextType:
        """Classify the context type of detected text."""
        text_lower = text.lower().strip()
        
        # Check if it's a common label
        if pii_type in self.common_labels:
            labels = self.common_labels[pii_type]
            if any(label in text_lower for label in labels):
                return ContextType.LABEL
        
        # Check if it's an instruction
        for pattern in self.instruction_regex:
            if pattern.search(text_lower):
                return ContextType.INSTRUCTION
        
        # Check if it matches actual PII patterns
        if pii_type in self.actual_pii_indicators:
            patterns = self.actual_pii_indicators[pii_type]
            for pattern_str in patterns:
                if re.search(pattern_str, text, re.IGNORECASE):
                    return ContextType.ACTUAL_PII
        
        # Default classification based on text characteristics
        if self._is_likely_label(text_lower):
            return ContextType.LABEL
        elif self._is_likely_heading(text):
            return ContextType.HEADING
        else:
            return ContextType.ACTUAL_PII

    def _is_likely_label(self, text: str) -> bool:
        """Check if text is likely a form label."""
        # Labels often end with colon
        if text.endswith(':'):
            return True
        
        # Labels are typically short (1-3 words)
        words = text.split()
        if len(words) <= 3:
            # Common label patterns
            label_indicators = [
                'name', 'email', 'phone', 'mobile', 'contact', 'address',
                'date', 'birth', 'age', 'gender', 'occupation', 'designation',
                'company', 'organization', 'department', 'id', 'number', 'code'
            ]
            if any(indicator in text for indicator in label_indicators):
                return True
        
        return False

    def _is_likely_heading(self, text: str) -> bool:
        """Check if text is likely a document heading."""
        # Headings are often in ALL CAPS or Title Case
        if text.isupper() or text.istitle():
            # Common heading words
            heading_words = [
                'personal', 'information', 'details', 'contact', 'address',
                'application', 'form', 'registration', 'profile', 'account'
            ]
            text_lower = text.lower()
            if any(word in text_lower for word in heading_words):
                return True
        
        return False

    def should_redact(self, detection: DetectionResult, surrounding_text: str = "") -> bool:
        """
        Determine if a detection should be redacted based on context.
        
        Args:
            detection: The PII detection result
            surrounding_text: Text around the detected PII for context
            
        Returns:
            True if should redact, False if it's likely a false positive
        """
        # Get the detected text
        detected_text = detection.text if hasattr(detection, 'text') else ""
        
        # Classify the context
        context = self.classify_context(detected_text, detection.pii_type)
        
        # Log the classification for debugging
        self.log_debug(f"Context classification for '{detected_text}' ({detection.pii_type}): {context}")
        
        # Don't redact labels, headings, or instructions
        if context in [ContextType.LABEL, ContextType.HEADING, ContextType.INSTRUCTION]:
            self.log_info(f"Skipping redaction of '{detected_text}' - classified as {context.value}")
            return False
        
        # Additional validation for specific PII types
        return self._validate_by_pii_type(detection, detected_text, surrounding_text)

    def _validate_by_pii_type(self, detection: DetectionResult, text: str, surrounding_text: str) -> bool:
        """Additional validation based on specific PII type."""
        pii_type = detection.pii_type
        
        if pii_type == PIIType.EMAIL:
            # Must contain @ symbol for valid email
            if '@' not in text:
                return False
            # Must have domain extension
            if not re.search(r'\.\w{2,}', text):
                return False
                
        elif pii_type == PIIType.PHONE:
            # Must contain enough digits
            digits = re.findall(r'\d', text)
            if len(digits) < 7:  # Minimum phone number length
                return False
                
        elif pii_type == PIIType.PERSON_NAME:
            # Avoid single words that are likely labels
            words = text.split()
            if len(words) == 1:
                # Single word - check if it's a common label
                if text.lower() in ['name', 'applicant', 'candidate', 'student', 'employee', 'customer']:
                    return False
            
            # Check if surrounded by form-like text
            if surrounding_text:
                form_indicators = ['enter', 'fill', 'provide', 'write', 'mention', ':', 'name of']
                if any(indicator in surrounding_text.lower() for indicator in form_indicators):
                    # This might be in a form context - be more careful
                    if len(words) == 1 or text.lower() in self.common_labels[PIIType.PERSON_NAME]:
                        return False
        
        return True

    def filter_detections(
        self, 
        detections: List[DetectionResult], 
        full_text: str = ""
    ) -> List[DetectionResult]:
        """
        Filter a list of detections to remove false positives.
        
        Args:
            detections: List of PII detections
            full_text: Full OCR text for context analysis
            
        Returns:
            Filtered list of detections
        """
        filtered_detections = []
        
        for detection in detections:
            # Get surrounding text for context (50 chars before and after)
            surrounding_text = ""
            if full_text and hasattr(detection, 'text'):
                try:
                    text_pos = full_text.find(detection.text)
                    if text_pos >= 0:
                        start = max(0, text_pos - 50)
                        end = min(len(full_text), text_pos + len(detection.text) + 50)
                        surrounding_text = full_text[start:end]
                except:
                    pass
            
            # Check if should redact
            if self.should_redact(detection, surrounding_text):
                filtered_detections.append(detection)
            else:
                self.log_info(f"Filtered out false positive: '{detection.text}' ({detection.pii_type})")
        
        self.log_info(f"Context filter: {len(detections)} → {len(filtered_detections)} detections")
        return filtered_detections


# Convenience function for easy integration
def filter_false_positives(detections: List[DetectionResult], full_text: str = "") -> List[DetectionResult]:
    """Convenience function to filter false positives from PII detections."""
    filter_instance = PIIContextFilter()
    return filter_instance.filter_detections(detections, full_text)
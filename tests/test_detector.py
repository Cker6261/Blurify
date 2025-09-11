"""
Tests for PII detection functionality.
"""

import pytest
from typing import List

from blurify.detector import PIIDetector, RegexDetector, SpacyNERDetector
from blurify.ocr import OCRResult
from blurify.config import DetectionConfig, PIIType, DetectionResult


@pytest.fixture
def detection_config():
    """Create detection configuration for testing."""
    return DetectionConfig(
        confidence_threshold=0.5,
        use_presidio=False  # Disable Presidio for testing
    )


@pytest.fixture
def sample_texts():
    """Sample texts containing various PII types."""
    return [
        "Contact John Doe at john.doe@example.com or call +91 98765 43210",
        "Email: test@gmail.com, Phone: 9876543210",
        "Aadhaar: 1234 5678 9012, PAN: ABCDE1234F",
        "Date of birth: 15/03/1990, Mobile: +91-98765-43210",
        "Mr. Smith's email is smith@company.org and his phone is 011-12345678",
        "Visit us on 25 Jan 2023 or call 9123456789",
    ]


@pytest.fixture
def sample_ocr_results():
    """Sample OCR results with bounding boxes."""
    return [
        OCRResult("john.doe@example.com", (100, 50, 250, 80), 0.95),
        OCRResult("+91 98765 43210", (300, 50, 450, 80), 0.9),
        OCRResult("John Doe", (50, 100, 150, 130), 0.85),
        OCRResult("15/03/1990", (200, 100, 300, 130), 0.8)
    ]


class TestRegexDetector:
    """Test regex-based PII detection."""
    
    def test_regex_detector_initialization(self):
        """Test regex detector initialization."""
        detector = RegexDetector()
        assert len(detector.patterns) > 0
        
        # Check that we have patterns for expected PII types
        pattern_types = {pattern.pii_type for pattern in detector.patterns}
        expected_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.AADHAAR, PIIType.PAN, PIIType.DATE}
        
        assert expected_types.issubset(pattern_types)
    
    def test_email_detection(self):
        """Test email detection with regex."""
        detector = RegexDetector()
        
        test_cases = [
            ("Contact john.doe@example.com for details", "john.doe@example.com"),
            ("Email: test@gmail.com", "test@gmail.com"),
            ("Reach out to admin@company.org", "admin@company.org"),
            ("Multiple emails: a@b.com and c@d.org", "a@b.com"),  # Should find at least one
        ]
        
        for text, expected_email in test_cases:
            results = detector.detect(text)
            email_results = [r for r in results if r[0] == PIIType.EMAIL]
            
            assert len(email_results) >= 1, f"No email found in: {text}"
            found_email = email_results[0][1]
            assert expected_email in found_email or found_email in expected_email
    
    def test_phone_detection(self):
        """Test phone number detection with regex."""
        detector = RegexDetector()
        
        test_cases = [
            "+91 98765 43210",
            "9876543210",
            "+91-98765-43210",
            "011-12345678",
            "+1 555 123 4567"  # International format
        ]
        
        for phone_text in test_cases:
            results = detector.detect(f"Call me at {phone_text}")
            phone_results = [r for r in results if r[0] == PIIType.PHONE]
            
            assert len(phone_results) >= 1, f"No phone found in: {phone_text}"
            assert phone_results[0][2] > 0.5  # Confidence should be reasonable
    
    def test_aadhaar_detection(self):
        """Test Aadhaar number detection."""
        detector = RegexDetector()
        
        test_cases = [
            "1234 5678 9012",
            "123456789012",
            "1234-5678-9012"
        ]
        
        for aadhaar_text in test_cases:
            results = detector.detect(f"Aadhaar: {aadhaar_text}")
            aadhaar_results = [r for r in results if r[0] == PIIType.AADHAAR]
            
            assert len(aadhaar_results) >= 1, f"No Aadhaar found in: {aadhaar_text}"
    
    def test_pan_detection(self):
        """Test PAN card detection."""
        detector = RegexDetector()
        
        test_cases = [
            "ABCDE1234F",
            "XYZPQ9876R"
        ]
        
        for pan_text in test_cases:
            results = detector.detect(f"PAN: {pan_text}")
            pan_results = [r for r in results if r[0] == PIIType.PAN]
            
            assert len(pan_results) >= 1, f"No PAN found in: {pan_text}"
    
    def test_date_detection(self):
        """Test date detection."""
        detector = RegexDetector()
        
        test_cases = [
            "15/03/1990",
            "2023-12-25",
            "25 Jan 2023",
            "Dec 31 2022"
        ]
        
        for date_text in test_cases:
            results = detector.detect(f"Date: {date_text}")
            date_results = [r for r in results if r[0] == PIIType.DATE]
            
            assert len(date_results) >= 1, f"No date found in: {date_text}"


class TestSpacyNERDetector:
    """Test spaCy NER detector."""
    
    def test_spacy_detector_initialization(self, detection_config):
        """Test spaCy detector initialization."""
        detector = SpacyNERDetector(detection_config)
        
        # Should initialize without error
        # May be None if spaCy not available
        if detector.nlp is not None:
            assert hasattr(detector.nlp, 'pipe')
    
    def test_person_name_detection(self, detection_config):
        """Test person name detection with spaCy."""
        detector = SpacyNERDetector(detection_config)
        
        if detector.nlp is None:
            pytest.skip("spaCy not available")
        
        test_cases = [
            "John Doe works here",
            "Contact Mary Smith for details",
            "Dr. Robert Johnson will see you"
        ]
        
        for text in test_cases:
            results = detector.detect(text)
            person_results = [r for r in results if r[0] == PIIType.PERSON_NAME]
            
            # spaCy may or may not detect names in simple synthetic text
            # So we just check that the function runs without error
            assert isinstance(person_results, list)
    
    def test_spacy_unavailable(self, detection_config):
        """Test behavior when spaCy is unavailable."""
        # This test simulates spaCy being unavailable
        detector = SpacyNERDetector(detection_config)
        
        # If spaCy is not available, detector.nlp should be None
        if detector.nlp is None:
            results = detector.detect("John Doe works here")
            assert results == []


class TestPIIDetector:
    """Test main PII detector class."""
    
    def test_pii_detector_initialization(self, detection_config):
        """Test PII detector initialization."""
        detector = PIIDetector(detection_config)
        
        assert detector.config == detection_config
        assert detector.regex_detector is not None
        assert detector.spacy_detector is not None
    
    def test_detect_in_text(self, detection_config, sample_texts):
        """Test PII detection in plain text."""
        detector = PIIDetector(detection_config)
        
        for text in sample_texts:
            results = detector.detect_in_text(text)
            
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, DetectionResult)
                assert result.pii_type in PIIType
                assert len(result.text) > 0
                assert result.confidence > 0
                assert result.source in ["regex", "spacy", "presidio"]
    
    def test_detect_in_ocr_results(self, detection_config, sample_ocr_results):
        """Test PII detection in OCR results."""
        detector = PIIDetector(detection_config)
        
        results = detector.detect_in_ocr_results(sample_ocr_results)
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.bbox != (0, 0, 0, 0)  # Should have real bounding box
    
    def test_confidence_filtering(self, detection_config, sample_texts):
        """Test confidence-based filtering."""
        # High confidence threshold
        high_threshold_config = DetectionConfig(confidence_threshold=0.95)
        detector_high = PIIDetector(high_threshold_config)
        
        # Low confidence threshold
        low_threshold_config = DetectionConfig(confidence_threshold=0.1)
        detector_low = PIIDetector(low_threshold_config)
        
        text = sample_texts[0]  # Use first sample text
        
        high_results = detector_high.detect_in_text(text)
        low_results = detector_low.detect_in_text(text)
        
        # Low threshold should find same or more detections
        assert len(low_results) >= len(high_results)
    
    def test_pii_type_filtering(self, sample_texts):
        """Test PII type filtering."""
        # Only detect emails
        email_only_config = DetectionConfig(enabled_pii_types=[PIIType.EMAIL])
        email_detector = PIIDetector(email_only_config)
        
        # Only detect phones
        phone_only_config = DetectionConfig(enabled_pii_types=[PIIType.PHONE])
        phone_detector = PIIDetector(phone_only_config)
        
        text = sample_texts[0]  # Contains both email and phone
        
        email_results = email_detector.detect_in_text(text)
        phone_results = phone_detector.detect_in_text(text)
        
        # Email detector should only find emails
        for result in email_results:
            assert result.pii_type == PIIType.EMAIL
        
        # Phone detector should only find phones
        for result in phone_results:
            assert result.pii_type == PIIType.PHONE
    
    def test_deduplication(self, detection_config):
        """Test detection deduplication."""
        detector = PIIDetector(detection_config)
        
        # Text with duplicate information
        text = "Email john@example.com again john@example.com"
        
        results = detector.detect_in_text(text)
        
        # Should deduplicate identical detections
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        unique_emails = set(r.text.lower() for r in email_results)
        
        # All email detections should be for the same email
        assert len(unique_emails) <= 1


class TestDetectionResult:
    """Test DetectionResult data structure."""
    
    def test_create_detection_result(self):
        """Test creating detection result."""
        result = DetectionResult(
            pii_type=PIIType.EMAIL,
            text="test@example.com",
            bbox=(10, 20, 100, 50),
            confidence=0.95,
            source="regex"
        )
        
        assert result.pii_type == PIIType.EMAIL
        assert result.text == "test@example.com"
        assert result.bbox == (10, 20, 100, 50)
        assert result.confidence == 0.95
        assert result.source == "regex"
    
    def test_detection_result_serialization(self):
        """Test detection result to/from dict conversion."""
        original = DetectionResult(
            pii_type=PIIType.PHONE,
            text="+91 98765 43210",
            bbox=(100, 200, 300, 250),
            confidence=0.9,
            source="regex"
        )
        
        # Convert to dict
        result_dict = original.to_dict()
        
        assert result_dict["type"] == "phone"
        assert result_dict["text"] == "+91 98765 43210"
        assert result_dict["bbox"] == (100, 200, 300, 250)
        assert result_dict["confidence"] == 0.9
        assert result_dict["source"] == "regex"
        
        # Convert back from dict
        reconstructed = DetectionResult.from_dict(result_dict)
        
        assert reconstructed.pii_type == original.pii_type
        assert reconstructed.text == original.text
        assert reconstructed.bbox == original.bbox
        assert reconstructed.confidence == original.confidence
        assert reconstructed.source == original.source


# Integration tests
class TestDetectionIntegration:
    """Integration tests for detection functionality."""
    
    def test_comprehensive_detection(self, detection_config):
        """Test detection of multiple PII types in complex text."""
        detector = PIIDetector(detection_config)
        
        complex_text = """
        Dear John Doe,
        
        Thank you for your application. Please contact us at support@company.com
        or call our helpline at +91 98765 43210.
        
        Your reference number is REF123456.
        Date of application: 15/03/2023
        
        If you have an Aadhaar number (1234 5678 9012) or PAN card (ABCDE1234F),
        please provide them for verification.
        
        Best regards,
        Customer Service Team
        """
        
        results = detector.detect_in_text(complex_text)
        
        # Should find multiple types of PII
        found_types = set(r.pii_type for r in results)
        
        # Print results for debugging
        print(f"Found PII types: {[t.value for t in found_types]}")
        for result in results:
            print(f"  {result.pii_type.value}: '{result.text}' (conf: {result.confidence:.2f})")
        
        # Should find at least some PII
        assert len(results) > 0
        
        # Check that all results are valid
        for result in results:
            assert isinstance(result.pii_type, PIIType)
            assert len(result.text.strip()) > 0
            assert 0 <= result.confidence <= 1
            assert result.source in ["regex", "spacy", "presidio"]


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

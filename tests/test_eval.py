"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from typing import List

from blurify.config import PIIType, DetectionResult
from eval.evaluation import (
    calculate_iou, compute_iou, detection_metrics, pixel_completeness,
    create_mask_from_detections, match_detections, evaluate_detections
)


class TestIoU:
    """Test IoU calculation functions."""
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 30, 30)
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        bbox1 = (10, 10, 20, 20)
        bbox2 = (10, 10, 20, 20)
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 1.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        bbox1 = (0, 0, 10, 10)   # Area = 100
        bbox2 = (5, 5, 15, 15)   # Area = 100
        
        # Intersection: (5,5) to (10,10) = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 = 1/7 ≈ 0.143
        
        iou = calculate_iou(bbox1, bbox2)
        expected_iou = 25.0 / 175.0
        assert abs(iou - expected_iou) < 1e-6
    
    def test_compute_iou_alias(self):
        """Test that compute_iou is an alias for calculate_iou."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        
        iou1 = calculate_iou(bbox1, bbox2)
        iou2 = compute_iou(bbox1, bbox2)
        
        assert iou1 == iou2
    
    def test_calculate_iou_edge_cases(self):
        """Test IoU with edge cases."""
        # Zero area boxes
        bbox1 = (0, 0, 0, 0)
        bbox2 = (0, 0, 10, 10)
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 0.0
        
        # Touching boxes (no intersection)
        bbox1 = (0, 0, 10, 10)
        bbox2 = (10, 10, 20, 20)
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 0.0


class TestDetectionMatching:
    """Test detection matching functionality."""
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections for testing."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test"),
            DetectionResult(PIIType.PHONE, "+91 12345 67890", (10, 50, 150, 70), 0.8, "test"),
            DetectionResult(PIIType.PERSON_NAME, "John Doe", (200, 10, 280, 30), 0.85, "test"),
        ]
        
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (12, 12, 98, 28), 1.0, "gt"),
            DetectionResult(PIIType.PHONE, "+91 12345 67890", (15, 55, 145, 75), 1.0, "gt"),
            DetectionResult(PIIType.DATE, "2023-01-01", (300, 10, 380, 30), 1.0, "gt"),
        ]
        
        return predicted, ground_truth
    
    def test_match_detections_perfect_match(self):
        """Test matching with perfect overlap."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 1.0, "gt")
        ]
        
        matches, unmatched_pred, unmatched_gt = match_detections(predicted, ground_truth, 0.5)
        
        assert len(matches) == 1
        assert len(unmatched_pred) == 0
        assert len(unmatched_gt) == 0
        assert matches[0] == (0, 0, 1.0)  # Perfect IoU
    
    def test_match_detections_no_match(self):
        """Test matching with no overlap."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.PHONE, "+91 12345", (200, 200, 300, 220), 1.0, "gt")
        ]
        
        matches, unmatched_pred, unmatched_gt = match_detections(predicted, ground_truth, 0.5)
        
        assert len(matches) == 0
        assert len(unmatched_pred) == 1
        assert len(unmatched_gt) == 1
        assert 0 in unmatched_pred
        assert 0 in unmatched_gt
    
    def test_match_detections_different_types(self):
        """Test that different PII types don't match."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.PHONE, "+91 12345", (10, 10, 100, 30), 1.0, "gt")  # Same bbox, different type
        ]
        
        matches, unmatched_pred, unmatched_gt = match_detections(predicted, ground_truth, 0.5)
        
        assert len(matches) == 0
        assert len(unmatched_pred) == 1
        assert len(unmatched_gt) == 1


class TestDetectionMetrics:
    """Test detection metrics calculation."""
    
    def test_detection_metrics_perfect_match(self):
        """Test metrics with perfect matching."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 1.0, "gt")
        ]
        
        metrics = detection_metrics(predicted, ground_truth)
        
        assert metrics["overall"]["precision"] == 1.0
        assert metrics["overall"]["recall"] == 1.0
        assert metrics["overall"]["f1"] == 1.0
        assert metrics["overall"]["tp"] == 1
        assert metrics["overall"]["fp"] == 0
        assert metrics["overall"]["fn"] == 0
    
    def test_detection_metrics_no_predictions(self):
        """Test metrics with no predictions."""
        predicted = []
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 1.0, "gt")
        ]
        
        metrics = detection_metrics(predicted, ground_truth)
        
        assert metrics["overall"]["precision"] == 0.0
        assert metrics["overall"]["recall"] == 0.0
        assert metrics["overall"]["f1"] == 0.0
        assert metrics["overall"]["tp"] == 0
        assert metrics["overall"]["fp"] == 0
        assert metrics["overall"]["fn"] == 1
    
    def test_detection_metrics_false_positives(self):
        """Test metrics with false positives."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test"),
            DetectionResult(PIIType.PHONE, "fake phone", (200, 200, 300, 220), 0.8, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 1.0, "gt")
        ]
        
        metrics = detection_metrics(predicted, ground_truth)
        
        assert metrics["overall"]["precision"] == 0.5  # 1 TP, 1 FP
        assert metrics["overall"]["recall"] == 1.0      # 1 TP, 0 FN
        assert abs(metrics["overall"]["f1"] - 2/3) < 1e-6  # 2*0.5*1.0/(0.5+1.0)
        assert metrics["overall"]["tp"] == 1
        assert metrics["overall"]["fp"] == 1
        assert metrics["overall"]["fn"] == 0
    
    def test_detection_metrics_per_type(self):
        """Test per-type metrics calculation."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test"),
            DetectionResult(PIIType.PHONE, "+91 12345", (10, 50, 150, 70), 0.8, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 1.0, "gt"),
            DetectionResult(PIIType.DATE, "2023-01-01", (300, 10, 380, 30), 1.0, "gt")
        ]
        
        metrics = detection_metrics(predicted, ground_truth)
        
        # Email: 1 TP, 0 FP, 0 FN
        assert metrics["per_type"]["email"]["precision"] == 1.0
        assert metrics["per_type"]["email"]["recall"] == 1.0
        assert metrics["per_type"]["email"]["f1"] == 1.0
        
        # Phone: 0 TP, 1 FP, 0 FN
        assert metrics["per_type"]["phone"]["precision"] == 0.0
        assert metrics["per_type"]["phone"]["recall"] == 0.0  # No ground truth phones
        assert metrics["per_type"]["phone"]["f1"] == 0.0
        
        # Date: 0 TP, 0 FP, 1 FN
        assert metrics["per_type"]["date"]["precision"] == 0.0  # No predicted dates
        assert metrics["per_type"]["date"]["recall"] == 0.0
        assert metrics["per_type"]["date"]["f1"] == 0.0
    
    def test_detection_metrics_empty_inputs(self):
        """Test metrics with empty inputs."""
        metrics = detection_metrics([], [])
        
        assert metrics["overall"]["precision"] == 1.0
        assert metrics["overall"]["recall"] == 1.0
        assert metrics["overall"]["f1"] == 1.0
        assert metrics["overall"]["tp"] == 0
        assert metrics["overall"]["fp"] == 0
        assert metrics["overall"]["fn"] == 0


class TestPixelMetrics:
    """Test pixel-level metrics."""
    
    def test_create_mask_from_detections(self):
        """Test mask creation from detections."""
        detections = [
            DetectionResult(PIIType.EMAIL, "test", (10, 10, 50, 30), 0.9, "test"),
            DetectionResult(PIIType.PHONE, "test", (100, 100, 150, 120), 0.8, "test")
        ]
        
        mask = create_mask_from_detections(detections, (200, 200))
        
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        
        # Check that detection regions are marked
        assert mask[10:30, 10:50].sum() > 0
        assert mask[100:120, 100:150].sum() > 0
        
        # Check that other regions are not marked
        assert mask[0, 0] == 0
        assert mask[199, 199] == 0
    
    def test_pixel_completeness_perfect_match(self):
        """Test pixel completeness with perfect match."""
        # Create identical masks
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:50] = 1
        mask[60:80, 60:90] = 1
        
        metrics = pixel_completeness(mask, mask)
        
        assert metrics["jaccard_index"] == 1.0
        assert metrics["pixel_precision"] == 1.0
        assert metrics["pixel_recall"] == 1.0
        assert metrics["pixel_f1"] == 1.0
        assert metrics["false_redaction_rate"] == 0.0
    
    def test_pixel_completeness_no_overlap(self):
        """Test pixel completeness with no overlap."""
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:30, 10:50] = 1
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[60:80, 60:90] = 1
        
        metrics = pixel_completeness(gt_mask, pred_mask)
        
        assert metrics["jaccard_index"] == 0.0
        assert metrics["pixel_precision"] == 0.0
        assert metrics["pixel_recall"] == 0.0
        assert metrics["pixel_f1"] == 0.0
        assert metrics["false_redaction_rate"] == 1.0  # All predicted pixels are false positives
    
    def test_pixel_completeness_partial_overlap(self):
        """Test pixel completeness with partial overlap."""
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:30, 10:50] = 1  # 20*40 = 800 pixels
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[15:35, 15:55] = 1  # 20*40 = 800 pixels
        
        # Intersection: [15:30, 15:50] = 15*35 = 525 pixels
        # Union: 800 + 800 - 525 = 1075 pixels
        
        metrics = pixel_completeness(gt_mask, pred_mask)
        
        expected_jaccard = 525.0 / 1075.0
        expected_precision = 525.0 / 800.0
        expected_recall = 525.0 / 800.0
        
        assert abs(metrics["jaccard_index"] - expected_jaccard) < 1e-6
        assert abs(metrics["pixel_precision"] - expected_precision) < 1e-6
        assert abs(metrics["pixel_recall"] - expected_recall) < 1e-6
    
    def test_pixel_completeness_empty_masks(self):
        """Test pixel completeness with empty masks."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        
        metrics = pixel_completeness(empty_mask, empty_mask)
        
        assert metrics["jaccard_index"] == 1.0  # Empty sets have Jaccard = 1
        assert metrics["pixel_precision"] == 0.0
        assert metrics["pixel_recall"] == 0.0
        assert metrics["pixel_f1"] == 0.0
        assert metrics["false_redaction_rate"] == 0.0


class TestEvaluationIntegration:
    """Integration tests for evaluation functionality."""
    
    def test_evaluate_detections_comprehensive(self):
        """Test comprehensive evaluation function."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test"),
            DetectionResult(PIIType.PHONE, "+91 12345", (10, 50, 150, 70), 0.8, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (12, 12, 98, 28), 1.0, "gt"),
            DetectionResult(PIIType.DATE, "2023-01-01", (300, 10, 380, 30), 1.0, "gt")
        ]
        
        results = evaluate_detections(
            predicted, ground_truth, 
            image_shape=(400, 200),  # height, width
            iou_threshold=0.5
        )
        
        # Should have detection metrics
        assert "detection_metrics" in results
        assert "overall" in results["detection_metrics"]
        assert "per_type" in results["detection_metrics"]
        
        # Should have pixel metrics
        assert "pixel_metrics" in results
        assert "jaccard_index" in results["pixel_metrics"]
        
        # Should have threshold info
        assert results["iou_threshold"] == 0.5
    
    def test_evaluate_detections_no_image_shape(self):
        """Test evaluation without image shape (no pixel metrics)."""
        predicted = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (10, 10, 100, 30), 0.9, "test")
        ]
        ground_truth = [
            DetectionResult(PIIType.EMAIL, "test@example.com", (12, 12, 98, 28), 1.0, "gt")
        ]
        
        results = evaluate_detections(predicted, ground_truth)
        
        # Should have detection metrics but no pixel metrics
        assert "detection_metrics" in results
        assert "pixel_metrics" not in results


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

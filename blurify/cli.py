"""
Command-line interface for Blurify.

Provides CLI access to all Blurify functionality with argparse.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union
import json
import time
from datetime import datetime

from .config import BlurifyConfig, RedactionMode, PIIType
from .ocr import OCRManager
from .detector import PIIDetector
from .visual_detector import VisualDetector
from .redactor import Redactor
from .pdf_utils import PDFConverter

def setup_d_drive_cache():
    """Set up D: drive cache directories for ML models."""
    import os
    from pathlib import Path
    
    d_cache_root = Path("D:/ml_cache")
    d_cache_root.mkdir(exist_ok=True)
    
    easyocr_cache = d_cache_root / "easyocr"
    easyocr_cache.mkdir(exist_ok=True)
    
    torch_cache = d_cache_root / "torch"
    torch_cache.mkdir(exist_ok=True)
    
    temp_dir = d_cache_root / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    os.environ['EASYOCR_MODULE_PATH'] = str(easyocr_cache)
    os.environ['TORCH_HOME'] = str(torch_cache)
    os.environ['TMPDIR'] = str(temp_dir)
    os.environ['TEMP'] = str(temp_dir)
    os.environ['TMP'] = str(temp_dir)
    
    print(f"✅ Using D: drive for ML cache: {d_cache_root}")
    return d_cache_root
from .logger import setup_root_logger, get_logger


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="blurify",
        description="Local, privacy-first PII redaction for scanned images and PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Redact single image with blur
  python -m blurify.cli --input demo_data/sample1.jpg --mode blur --output out/

  # Redact PDF with masking
  python -m blurify.cli --input document.pdf --mode mask --output redacted/

  # Batch process directory
  python -m blurify.cli --input images/ --output redacted/ --recursive

  # Custom configuration
  python -m blurify.cli --input file.jpg --config custom_config.json

  # Evaluation mode
  python -m blurify.cli --eval --input demo_data/ --ground-truth demo_data/ground_truth.json
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file or directory path"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory path (default: output)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["blur", "mask"],
        default="blur",
        help="Redaction mode: blur or mask (default: blur)"
    )
    
    # Processing options
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"],
        help="File extensions to process (default: common image/PDF formats)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["easyocr", "tesseract"],
        help="OCR engine to use (overrides config)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Minimum confidence threshold for detections (0.0-1.0)"
    )
    
    parser.add_argument(
        "--enable-visual",
        action="store_true",
        help="Enable visual detection (faces, signatures, photos)"
    )
    
    parser.add_argument(
        "--pii-types",
        type=str,
        nargs="+",
        choices=["email", "phone", "person_name", "date", "aadhaar", "pan", "signature", "photo"],
        help="PII types to detect (default: all)"
    )
    
    # Evaluation mode
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode"
    )
    
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth JSON file (for evaluation)"
    )
    
    # Output options
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        default=True,
        help="Save processing metadata as JSON (default: True)"
    )
    
    parser.add_argument(
        "--preserve-original",
        action="store_true",
        default=True,
        help="Keep original files (default: True)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["same", "png", "jpg", "pdf"],
        default="same",
        help="Output format (default: same as input)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (default: console only)"
    )
    
    # Misc
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Blurify 0.1.0"
    )
    
    return parser


def load_config(config_path: Optional[str] = None) -> BlurifyConfig:
    """Load configuration from file or use defaults."""
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # TODO: Implement proper config loading from JSON
        # For now, use default config
        config = BlurifyConfig()
    else:
        config = BlurifyConfig()
    
    return config


def find_input_files(
    input_path: Union[str, Path],
    extensions: List[str],
    recursive: bool = False
) -> List[Path]:
    """Find input files based on path and extensions."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in [ext.lower() for ext in extensions]:
            files.append(input_path)
    elif input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        for ext in extensions:
            files.extend(input_path.glob(f"{pattern}{ext}"))
            files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(list(set(files)))


def process_single_file(
    input_path: Path,
    output_dir: Path,
    config: BlurifyConfig,
    mode: RedactionMode,
    ocr_manager: OCRManager,
    pii_detector: PIIDetector,
    visual_detector: Optional[VisualDetector],
    redactor: Redactor,
    pdf_converter: Optional[PDFConverter],
    logger
) -> dict:
    """Process a single input file."""
    logger.info(f"Processing: {input_path}")
    
    start_time = time.time()
    processing_info = {
        "input_file": str(input_path),
        "success": False,
        "error": None,
        "processing_time_ms": 0,
        "detections": 0
    }
    
    try:
        # Determine if input is PDF
        is_pdf = input_path.suffix.lower() == '.pdf'
        
        if is_pdf:
            if pdf_converter is None:
                raise RuntimeError("PDF processing requires PDF converter")
            
            # Convert PDF to images
            temp_images = pdf_converter.pdf_to_images(input_path, dpi=200)
            process_images = temp_images
        else:
            process_images = [input_path]
        
        all_detections = []
        processed_images = []
        
        for image_path in process_images:
            # OCR extraction
            ocr_results, engine_used = ocr_manager.extract_text(image_path)
            logger.debug(f"OCR found {len(ocr_results)} text regions using {engine_used}")
            
            # PII detection
            text_detections = pii_detector.detect_in_ocr_results(ocr_results)
            
            # Visual detection (if enabled)
            visual_detections = []
            if visual_detector:
                visual_detections = visual_detector.detect_visual_elements(image_path)
            
            # Combine detections
            image_detections = text_detections + visual_detections
            all_detections.extend(image_detections)
            
            # Apply redaction
            if image_detections:
                output_filename = f"redacted_{image_path.name}"
                output_path = output_dir / output_filename
                
                redacted_image, redaction_metadata = redactor.redact_image(
                    image_path, image_detections, mode, output_path
                )
                processed_images.append(output_path)
                
                logger.info(f"Redacted {len(image_detections)} regions in {image_path.name}")
            else:
                # No detections, copy original if preserve_original is False
                if not config.preserve_original:
                    output_path = output_dir / image_path.name
                    import shutil
                    shutil.copy2(image_path, output_path)
                    processed_images.append(output_path)
                
                logger.info(f"No PII detected in {image_path.name}")
        
        # If input was PDF, convert processed images back to PDF
        if is_pdf and processed_images:
            output_pdf_path = output_dir / f"redacted_{input_path.name}"
            pdf_converter.images_to_pdf(processed_images, output_pdf_path)
            
            # Clean up temporary images
            for temp_image in temp_images:
                temp_image.unlink(missing_ok=True)
            for proc_image in processed_images:
                proc_image.unlink(missing_ok=True)
        
        # Save metadata
        if config.save_metadata:
            metadata_path = output_dir / f"{input_path.stem}_metadata.json"
            metadata = {
                "input_file": str(input_path),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "detections": [d.to_dict() for d in all_detections],
                "redaction_mode": mode.value,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        processing_info.update({
            "success": True,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "detections": len(all_detections)
        })
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        processing_info["error"] = str(e)
    
    return processing_info


def run_evaluation(
    input_path: Path,
    ground_truth_path: Path,
    config: BlurifyConfig,
    logger
) -> dict:
    """Run evaluation against ground truth."""
    logger.info("Running evaluation mode")
    
    # Import evaluation module
    try:
        from ..eval.evaluation import evaluate_detections
    except ImportError:
        logger.error("Evaluation module not available")
        return {"error": "Evaluation module not available"}
    
    # Load ground truth
    try:
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
        return {"error": f"Failed to load ground truth: {e}"}
    
    # TODO: Implement evaluation logic
    logger.warning("Evaluation mode is not fully implemented yet")
    return {"error": "Evaluation not implemented"}


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_root_logger(args.log_level, log_file)
    logger = get_logger(__name__)
    
    logger.info("Blurify CLI started")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with CLI arguments
        if args.ocr_engine:
            config.ocr.primary_engine = args.ocr_engine
        
        if args.confidence_threshold is not None:
            config.detection.confidence_threshold = args.confidence_threshold
        
        if args.pii_types:
            config.detection.enabled_pii_types = [PIIType(t) for t in args.pii_types]
        
        # Set redaction mode
        mode = RedactionMode(args.mode)
        config.redaction.default_mode = mode
        
        # Setup output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        input_files = find_input_files(
            args.input, args.extensions, args.recursive
        )
        
        if not input_files:
            logger.error("No input files found")
            return 1
        
        logger.info(f"Found {len(input_files)} files to process")
        
        if args.dry_run:
            logger.info("DRY RUN - Files that would be processed:")
            for file_path in input_files:
                logger.info(f"  {file_path}")
            return 0
        
        # Handle evaluation mode
        if args.eval:
            if not args.ground_truth:
                logger.error("Evaluation mode requires --ground-truth argument")
                return 1
            
            result = run_evaluation(
                Path(args.input),
                Path(args.ground_truth),
                config,
                logger
            )
            
            if "error" in result:
                logger.error(f"Evaluation failed: {result['error']}")
                return 1
            
            logger.info("Evaluation completed successfully")
            return 0
        
        # Initialize processing components
        logger.info("Initializing processing components...")
        
        try:
            ocr_manager = OCRManager(config.ocr)
            logger.info(f"OCR engines available: {ocr_manager.get_available_engines()}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            return 1
        
        try:
            pii_detector = PIIDetector(config.detection)
        except Exception as e:
            logger.error(f"Failed to initialize PII detector: {e}")
            return 1
        
        visual_detector = None
        if args.enable_visual:
            try:
                visual_detector = VisualDetector(config.visual_detection)
                logger.info("Visual detector enabled")
            except Exception as e:
                logger.warning(f"Visual detector initialization failed: {e}")
        
        redactor = Redactor(config.redaction)
        
        # Initialize PDF converter if needed
        pdf_converter = None
        if any(f.suffix.lower() == '.pdf' for f in input_files):
            try:
                pdf_converter = PDFConverter(config)
            except Exception as e:
                logger.error(f"Failed to initialize PDF converter: {e}")
                logger.error("PDF files cannot be processed")
        
        # Process files
        results = []
        for i, input_file in enumerate(input_files, 1):
            logger.info(f"Processing file {i}/{len(input_files)}: {input_file.name}")
            
            result = process_single_file(
                input_file, output_dir, config, mode,
                ocr_manager, pii_detector, visual_detector, redactor,
                pdf_converter, logger
            )
            results.append(result)
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        total_detections = sum(r["detections"] for r in results if r["success"])
        avg_time = sum(r["processing_time_ms"] for r in results if r["success"]) / max(successful, 1)
        
        logger.info(f"""
Processing complete!
  Total files: {len(results)}
  Successful: {successful}
  Failed: {failed}
  Total PII detections: {total_detections}
  Average processing time: {avg_time:.2f}ms
  Output directory: {output_dir}
        """)
        
        if failed > 0:
            logger.warning(f"{failed} files failed to process")
            for result in results:
                if not result["success"]:
                    logger.warning(f"  {result['input_file']}: {result['error']}")
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Set up D: drive cache before running main
    setup_d_drive_cache()
    sys.exit(main())

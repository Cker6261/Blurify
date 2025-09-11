"""
Streamlit web UI for Blurify PII redaction tool.

Provides an easy-to-use interface for uploading files and redacting PII.
"""

import streamlit as st
import tempfile
import json
import os
from pathlib import Path
from PIL import Image
import io
import time
from typing import List, Optional

# Setup D: drive cache before importing Blurify
import os
from pathlib import Path

def setup_d_drive_cache():
    """Set up D: drive cache directories for ML models."""
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

# Setup cache before imports
setup_d_drive_cache()

# Import Blurify components
try:
    from blurify.config import BlurifyConfig, RedactionMode, PIIType
    from blurify.ocr import OCRManager
    from blurify.detector import PIIDetector
    from blurify.visual_detector import VisualDetector
    from blurify.redactor import Redactor
    from blurify.pdf_utils import PDFConverter
    from blurify.logger import setup_root_logger, get_logger
except ImportError:
    st.error("""
    Blurify package not found. Please install it first:
    ```
    pip install -e .
    ```
    """)
    st.stop()


def setup_page_config():
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="Blurify - PII Redaction Tool",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Show cache setup info
    d_cache_path = Path("D:/ml_cache")
    if d_cache_path.exists():
        st.success(f"✅ Using D: drive cache: {d_cache_path}")
    else:
        st.warning("⚠️ D: drive cache not found, creating...")
    
    # Debug controls
    with st.expander("🔧 Debug Controls"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Components"):
                if 'components' in st.session_state:
                    del st.session_state.components
                st.success("Components will be reinitialized on next run")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Results"):
                if 'results' in st.session_state:
                    del st.session_state.results
                st.success("Results cleared")
                st.rerun()
        
        # System info
        import platform
        st.info(f"System: {platform.system()} {platform.release()}")
        
        # Show current session state size
        session_keys = list(st.session_state.keys())
        st.info(f"Session state keys: {len(session_keys)}")


def initialize_components(config: BlurifyConfig) -> tuple:
    """Initialize Blurify components."""
    with st.spinner("Initializing components..."):
        try:
            # Initialize OCR
            ocr_manager = OCRManager(config.ocr)
            st.success(f"✅ OCR initialized: {', '.join(ocr_manager.get_available_engines())}")
            
            # Initialize PII detector
            pii_detector = PIIDetector(config.detection)
            st.success("✅ PII detector initialized")
            
            # Initialize visual detector (optional)
            visual_detector = None
            if config.visual_detection.enabled_types:
                try:
                    visual_detector = VisualDetector(config.visual_detection)
                    st.success("✅ Visual detector initialized")
                except Exception as e:
                    st.warning(f"⚠️ Visual detector failed: {e}")
            
            # Initialize redactor
            redactor = Redactor(config.redaction)
            st.success("✅ Redactor initialized")
            
            # Initialize PDF converter (optional)
            pdf_converter = None
            try:
                pdf_converter = PDFConverter(config)
                st.success("✅ PDF converter initialized")
            except Exception as e:
                st.warning(f"⚠️ PDF converter failed: {e}")
            
            return ocr_manager, pii_detector, visual_detector, redactor, pdf_converter
            
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            import traceback
            st.error(f"Debug traceback: {traceback.format_exc()}")
            return None, None, None, None, None


def create_sidebar_config() -> BlurifyConfig:
    """Create configuration from sidebar inputs."""
    st.sidebar.header("🔧 Configuration")
    
    # Redaction mode
    mode = st.sidebar.selectbox(
        "Redaction Mode",
        options=["blur", "mask"],
        help="Choose how to redact detected PII"
    )
    
    # PII types to detect
    st.sidebar.subheader("PII Types to Detect")
    pii_types = []
    
    if st.sidebar.checkbox("📧 Email addresses", value=True):
        pii_types.append(PIIType.EMAIL)
    if st.sidebar.checkbox("📱 Phone numbers", value=True):
        pii_types.append(PIIType.PHONE)
    if st.sidebar.checkbox("👤 Person names", value=True):
        pii_types.append(PIIType.PERSON_NAME)
    if st.sidebar.checkbox("📅 Dates", value=True):
        pii_types.append(PIIType.DATE)
    if st.sidebar.checkbox("🆔 Aadhaar numbers", value=True):
        pii_types.append(PIIType.AADHAAR)
    if st.sidebar.checkbox("💳 PAN numbers", value=True):
        pii_types.append(PIIType.PAN)
    if st.sidebar.checkbox("✍️ Signatures", value=False):
        pii_types.append(PIIType.SIGNATURE)
    if st.sidebar.checkbox("📷 Photos/Faces", value=False):
        pii_types.append(PIIType.PHOTO)
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence for PII detection"
        )
        
        ocr_engine = st.selectbox(
            "OCR Engine",
            options=["easyocr", "tesseract"],
            help="Primary OCR engine to use"
        )
        
        blur_strength = st.slider(
            "Blur Strength",
            min_value=5,
            max_value=25,
            value=15,
            step=2,
            help="Gaussian blur kernel size"
        )
        
        enable_visual = st.checkbox(
            "Enable Visual Detection",
            value=False,
            help="Detect faces, signatures, and photos (experimental)"
        )
    
    # Create configuration
    config = BlurifyConfig()
    
    # Apply settings
    config.redaction.default_mode = RedactionMode(mode)
    config.detection.enabled_pii_types = pii_types
    config.detection.confidence_threshold = confidence_threshold
    config.ocr.primary_engine = ocr_engine
    config.redaction.blur_kernel_size = blur_strength
    
    if enable_visual:
        config.visual_detection.enabled_types = [PIIType.SIGNATURE, PIIType.PHOTO]
    else:
        config.visual_detection.enabled_types = []
    
    return config


def process_single_pdf_page(page_path, page_num, ocr_manager, pii_detector, visual_detector, redactor, config):
    """Process a single PDF page with memory-conscious approach."""
    try:
        # Resize image more aggressively for multi-page processing
        with Image.open(page_path) as img:
            original_size = img.size
            
            # Use smaller max dimension for multi-page to save memory
            max_dimension = 1000  # Smaller than single page processing
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                resized_img.save(page_path)
                st.info(f"📐 Page {page_num}: Resized from {original_size} to {new_size}")
        
        # OCR extraction
        try:
            ocr_results, engine_used = ocr_manager.extract_text(page_path)
            st.success(f"✅ Page {page_num}: OCR found {len(ocr_results)} text regions")
        except Exception as e:
            st.error(f"❌ Page {page_num}: OCR failed - {str(e)}")
            return None
        
        # PII detection
        try:
            text_detections = pii_detector.detect_in_ocr_results(ocr_results)
            st.success(f"✅ Page {page_num}: Found {len(text_detections)} text PII detections")
        except Exception as e:
            st.error(f"❌ Page {page_num}: PII detection failed - {str(e)}")
            return None
        
        # Visual detection (optional, skip if it causes issues)
        visual_detections = []
        if visual_detector and config.visual_detection.enabled_types:
            try:
                visual_detections = visual_detector.detect_visual_elements(page_path)
                st.success(f"✅ Page {page_num}: Visual detection completed")
            except Exception as e:
                st.warning(f"⚠️ Page {page_num}: Visual detection failed, continuing without it")
                visual_detections = []
        
        # Combine all detections
        all_detections = text_detections + visual_detections
        
        # Apply redaction if PII found
        redacted_image = None
        metadata = {}
        
        if all_detections:
            try:
                redacted_array, metadata = redactor.redact_image(
                    page_path, all_detections, config.redaction.default_mode
                )
                # Convert to PIL Image
                import cv2
                redacted_rgb = cv2.cvtColor(redacted_array, cv2.COLOR_BGR2RGB)
                redacted_image = Image.fromarray(redacted_rgb)
                st.success(f"✅ Page {page_num}: Redaction applied to {len(all_detections)} regions")
            except Exception as e:
                st.error(f"❌ Page {page_num}: Redaction failed - {str(e)}")
                return None
        
        # Load original image
        try:
            original_image = Image.open(page_path)
        except Exception as e:
            st.error(f"❌ Page {page_num}: Failed to load original image - {str(e)}")
            return None
        
        # Create page data
        page_data = {
            'page_number': page_num,
            'original_image': original_image,
            'redacted_image': redacted_image,
            'detections': all_detections,
            'ocr_results': ocr_results,
            'metadata': metadata,
            'engine_used': engine_used
        }
        
        return page_data
        
    except Exception as e:
        st.error(f"❌ Page {page_num}: Unexpected error - {str(e)}")
        return None


def process_uploaded_file(uploaded_file, components, config) -> Optional[tuple]:
    """Process uploaded file and return results."""
    ocr_manager, pii_detector, visual_detector, redactor, pdf_converter = components
    
    if any(comp is None for comp in [ocr_manager, pii_detector, redactor]):
        st.error("Some components failed to initialize. Cannot process file.")
        return None
    
    tmp_path = None
    image_paths = []
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)
        
        start_time = time.time()
        
        # Handle PDF files
        if tmp_path.suffix.lower() == '.pdf':
            if pdf_converter is None:
                st.error("PDF processing not available. Please upload an image file.")
                return None
            
            # Convert PDF to images
            with st.spinner("Converting PDF to images..."):
                try:
                    image_paths = pdf_converter.pdf_to_images(tmp_path)
                    st.success(f"✅ Converted PDF to {len(image_paths)} images")
                except Exception as e:
                    st.error(f"Failed to convert PDF: {str(e)}")
                    return None
            
            if not image_paths:
                st.error("Failed to convert PDF to images.")
                return None
            
            # Process all pages with proper memory management
            is_pdf = True
            all_pages_data = []
            total_detections = 0
            
            st.info(f"📄 Processing {len(image_paths)} pages from PDF...")
            
            # Process each page individually
            for page_idx, page_path in enumerate(image_paths):
                page_num = page_idx + 1
                
                try:
                    st.info(f"🔄 Processing page {page_num} of {len(image_paths)}...")
                    
                    # Process single page with memory-conscious approach
                    page_result = process_single_pdf_page(
                        page_path, page_num, ocr_manager, pii_detector, 
                        visual_detector, redactor, config
                    )
                    
                    if page_result:
                        all_pages_data.append(page_result)
                        total_detections += len(page_result['detections'])
                        st.success(f"✅ Page {page_num}: {len(page_result['detections'])} PII items found and redacted")
                    else:
                        st.warning(f"⚠️ Page {page_num}: Processing failed, skipping")
                    
                    # Force garbage collection after each page
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"❌ Page {page_num}: {str(e)}")
                    continue
            
            if not all_pages_data:
                st.error("Failed to process any pages from the PDF.")
                return None
            
            processing_time = time.time() - start_time
            
            st.success(f"✅ PDF processing complete: {len(all_pages_data)} pages processed, {total_detections} total PII items")
            
            # Return multi-page results
            return "multipage", all_pages_data, processing_time, {
                "total_pages": len(all_pages_data),
                "total_detections": total_detections,
                "image_paths": image_paths
            }
        else:
            # Single image processing
            process_path = tmp_path
            is_pdf = False
            image_paths = []
        
        # Verify image exists and is readable, resize if too large
        try:
            with Image.open(process_path) as test_img:
                st.success(f"✅ Image loaded: {test_img.size} pixels")
                
                # Resize image if too large to prevent memory issues
                max_dimension = 1600
                if max(test_img.size) > max_dimension:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_dimension / max(test_img.size)
                    new_size = tuple(int(dim * ratio) for dim in test_img.size)
                    
                    # Resize and save back
                    resized_img = test_img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_img.save(process_path)
                    st.info(f"📐 Resized image from {test_img.size} to {new_size} to optimize memory usage")
                    
        except Exception as e:
            st.error(f"Cannot read image file: {str(e)}")
            return None
        
        # OCR extraction
        with st.spinner("Performing OCR..."):
            try:
                ocr_results, engine_used = ocr_manager.extract_text(process_path)
                st.success(f"✅ OCR completed using {engine_used}. Found {len(ocr_results)} text regions.")
            except Exception as e:
                st.error(f"OCR failed: {str(e)}")
                return None
        
        # PII detection
        with st.spinner("Detecting PII..."):
            try:
                text_detections = pii_detector.detect_in_ocr_results(ocr_results)
                st.success(f"✅ Text PII detection completed. Found {len(text_detections)} detections.")
            except Exception as e:
                st.error(f"PII detection failed: {str(e)}")
                return None
        
        # Visual detection (if enabled)
        visual_detections = []
        if visual_detector and config.visual_detection.enabled_types:
            with st.spinner("Detecting visual elements..."):
                try:
                    visual_detections = visual_detector.detect_visual_elements(process_path)
                    st.success(f"✅ Visual detection completed. Found {len(visual_detections)} detections.")
                except Exception as e:
                    st.warning(f"⚠️ Visual detection failed (continuing): {str(e)}")
                    visual_detections = []
        
        # Combine detections
        all_detections = text_detections + visual_detections
        
        if not all_detections:
            st.info("No PII detected in the uploaded file.")
            # Still show original image
            try:
                original_image = Image.open(process_path)
                return "singlepage", original_image, None, time.time() - start_time, {
                    "detections": [],
                    "metadata": {},
                    "ocr_results": ocr_results,
                    "engine_used": engine_used
                }
            except Exception as e:
                st.error(f"Failed to load original image: {str(e)}")
                return None
        
        # Apply redaction
        with st.spinner("Applying redaction..."):
            try:
                redacted_image_array, metadata = redactor.redact_image(
                    process_path, all_detections, config.redaction.default_mode
                )
                st.success(f"✅ Redaction completed. Applied {len(all_detections)} redactions.")
            except Exception as e:
                st.error(f"Redaction failed: {str(e)}")
                return None
        
        # Convert to PIL Image for display
        try:
            import cv2
            redacted_image_rgb = cv2.cvtColor(redacted_image_array, cv2.COLOR_BGR2RGB)
            redacted_image = Image.fromarray(redacted_image_rgb)
            st.success("✅ Image conversion completed")
        except Exception as e:
            st.error(f"Image conversion failed: {str(e)}")
            return None
        
        # Load original image
        try:
            original_image = Image.open(process_path)
            st.success("✅ Original image loaded")
        except Exception as e:
            st.error(f"Failed to load original image: {str(e)}")
            return None
        
        processing_time = time.time() - start_time
        
        # Return single-page result in same format
        return "singlepage", original_image, redacted_image, processing_time, {
            "detections": all_detections,
            "metadata": metadata,
            "ocr_results": ocr_results,
            "engine_used": engine_used
        }
        
    except Exception as e:
        st.error(f"Processing failed with unexpected error: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None
        
    finally:
        # Cleanup files
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
                st.success("✅ Temporary file cleaned up")
        except Exception as e:
            st.warning(f"Failed to cleanup temp file: {str(e)}")
        
        try:
            for img_path in image_paths:
                if img_path and img_path.exists():
                    img_path.unlink(missing_ok=True)
            if image_paths:
                st.success(f"✅ {len(image_paths)} PDF image files cleaned up")
        except Exception as e:
            st.warning(f"Failed to cleanup PDF images: {str(e)}")


def display_results(original_image, redacted_image, processing_time, results_data):
    """Display processing results."""
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Original")
        try:
            st.image(original_image, use_column_width=True)
        except Exception as e:
            st.error(f"Failed to display original image: {str(e)}")
    
    with col2:
        st.subheader("🔒 Redacted")
        if redacted_image is None:
            st.info("No redaction applied - no PII detected")
        elif isinstance(redacted_image, Image.Image):
            try:
                st.image(redacted_image, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to display redacted image: {str(e)}")
        else:
            st.info("No redaction applied - no PII detected")
    
    # Processing stats
    detections = results_data.get("detections", [])
    
    st.subheader("📊 Processing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PII Detections", len(detections))
    
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col3:
        ocr_results = results_data.get("ocr_results", [])
        st.metric("Text Regions", len(ocr_results))
    
    with col4:
        engine_used = results_data.get("engine_used", "Unknown")
        st.metric("OCR Engine", engine_used)
    
    # Detection details
    if detections:
        st.subheader("🔍 Detection Details")
        
        # Group by PII type
        pii_counts = {}
        for detection in detections:
            pii_type = detection.pii_type.value
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
        
        # Display counts
        cols = st.columns(len(pii_counts))
        for i, (pii_type, count) in enumerate(pii_counts.items()):
            with cols[i]:
                st.metric(pii_type.title(), count)
        
        # Detailed detection table
        with st.expander("📋 Detailed Detection List"):
            detection_data = []
            for i, detection in enumerate(detections):
                detection_data.append({
                    "ID": i + 1,
                    "Type": detection.pii_type.value,
                    "Text": detection.text[:50] + "..." if len(detection.text) > 50 else detection.text,
                    "Confidence": f"{detection.confidence:.2f}",
                    "Source": detection.source,
                    "Bbox": f"{detection.bbox}"
                })
            
            st.dataframe(detection_data, use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if isinstance(redacted_image, Image.Image):
            try:
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                redacted_image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Redacted Image",
                    data=img_bytes,
                    file_name="redacted_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Failed to prepare download: {str(e)}")
        else:
            st.info("No redacted image to download")
    
    with col2:
        # Create metadata JSON
        metadata_json = {
            "processing_time_ms": processing_time * 1000,
            "total_detections": len(detections),
            "detections": [det.to_dict() for det in detections],
            "ocr_engine": engine_used,
            "ocr_regions": len(ocr_results)
        }
        
        metadata_bytes = json.dumps(metadata_json, indent=2).encode()
        
        st.download_button(
            label="📋 Download Metadata",
            data=metadata_bytes,
            file_name="redaction_metadata.json",
            mime="application/json"
        )


def display_multipage_results(all_pages_data, processing_time, summary_data):
    """Display results for multi-page PDF processing."""
    
    # Summary statistics
    st.subheader("📊 Multi-Page Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", summary_data['total_pages'])
    
    with col2:
        st.metric("Total PII Detections", summary_data['total_detections'])
    
    with col3:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col4:
        avg_time = processing_time / summary_data['total_pages']
        st.metric("Avg Time/Page", f"{avg_time:.2f}s")
    
    # Page selector
    st.subheader("📄 Page Results")
    
    # Create tabs for each page
    if len(all_pages_data) <= 5:
        # Use tabs for small number of pages
        tab_names = [f"Page {page['page_number']}" for page in all_pages_data]
        tabs = st.tabs(tab_names)
        
        for tab, page_data in zip(tabs, all_pages_data):
            with tab:
                display_single_page_result(page_data)
    else:
        # Use selectbox for many pages
        page_options = [f"Page {page['page_number']}" for page in all_pages_data]
        selected_page = st.selectbox("Select Page to View", page_options)
        
        # Find selected page data
        selected_idx = page_options.index(selected_page)
        page_data = all_pages_data[selected_idx]
        
        display_single_page_result(page_data)
    
    # Download all pages section
    st.subheader("💾 Download All Pages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create ZIP file with all redacted pages (pre-generate to avoid button state issues)
        try:
            import zipfile
            import tempfile
            
            zip_data = None
            redacted_pages = [p for p in all_pages_data if p['redacted_image']]
            
            if redacted_pages:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for page_data in redacted_pages:
                        # Convert to bytes
                        img_buffer = io.BytesIO()
                        page_data['redacted_image'].save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        # Add to ZIP
                        zf.writestr(f"page_{page_data['page_number']:03d}_redacted.png", img_bytes)
                
                zip_data = zip_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download All Redacted Pages as ZIP",
                    data=zip_data,
                    file_name="all_pages_redacted.zip",
                    mime="application/zip",
                    help=f"Download ZIP containing {len(redacted_pages)} redacted pages",
                    key="download_all_zip"
                )
            else:
                st.info("No redacted pages to download")
                
        except Exception as e:
            st.error(f"Failed to create ZIP file: {str(e)}")
            st.button("📥 Download All Redacted Pages as ZIP (Error)", disabled=True)
    
    with col2:
        # Create combined metadata for all pages
        combined_metadata = {
            "total_pages": summary_data['total_pages'],
            "total_processing_time_ms": processing_time * 1000,
            "total_detections": summary_data['total_detections'],
            "pages": []
        }
        
        for page_data in all_pages_data:
            page_metadata = {
                "page_number": page_data['page_number'],
                "detections": [det.to_dict() for det in page_data['detections']],
                "ocr_regions": len(page_data['ocr_results']),
                "ocr_engine": page_data['engine_used']
            }
            combined_metadata["pages"].append(page_metadata)
        
        metadata_bytes = json.dumps(combined_metadata, indent=2).encode()
        
        st.download_button(
            label="📋 Download Combined Metadata",
            data=metadata_bytes,
            file_name="all_pages_metadata.json",
            mime="application/json"
        )


def display_single_page_result(page_data):
    """Display results for a single page."""
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📄 Page {page_data['page_number']} - Original")
        try:
            st.image(page_data['original_image'], width=400)
        except Exception as e:
            st.error(f"Failed to display original image: {str(e)}")
    
    with col2:
        st.subheader(f"🔒 Page {page_data['page_number']} - Redacted")
        if page_data['redacted_image']:
            try:
                st.image(page_data['redacted_image'], width=400)
            except Exception as e:
                st.error(f"Failed to display redacted image: {str(e)}")
        else:
            st.info("No redaction applied - no PII detected")
    
    # Page statistics
    detections = page_data['detections']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PII Detections", len(detections))
    
    with col2:
        st.metric("Text Regions", len(page_data['ocr_results']))
    
    with col3:
        st.metric("OCR Engine", page_data['engine_used'])
    
    # Detection details for this page
    if detections:
        st.subheader(f"🔍 Page {page_data['page_number']} - Detection Details")
        
        # Group by PII type
        pii_counts = {}
        for detection in detections:
            pii_type = detection.pii_type.value
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
        
        # Display counts
        if pii_counts:
            cols = st.columns(len(pii_counts))
            for i, (pii_type, count) in enumerate(pii_counts.items()):
                with cols[i]:
                    st.metric(pii_type.title(), count)
        
        # Detailed detection table
        with st.expander(f"📋 Page {page_data['page_number']} - Detailed Detection List"):
            detection_data = []
            for i, detection in enumerate(detections):
                detection_data.append({
                    "ID": i + 1,
                    "Type": detection.pii_type.value,
                    "Text": detection.text[:50] + "..." if len(detection.text) > 50 else detection.text,
                    "Confidence": f"{detection.confidence:.2f}",
                    "Source": detection.source,
                    "Bbox": f"{detection.bbox}"
                })
            
            st.dataframe(detection_data, use_container_width=True)
    
    # Individual page download
    col1, col2 = st.columns(2)
    
    with col1:
        if page_data['redacted_image']:
            try:
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                page_data['redacted_image'].save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                st.download_button(
                    label=f"📥 Download Page {page_data['page_number']} Redacted",
                    data=img_bytes,
                    file_name=f"page_{page_data['page_number']:03d}_redacted.png",
                    mime="image/png",
                    key=f"download_page_{page_data['page_number']}"
                )
            except Exception as e:
                st.error(f"Failed to prepare download: {str(e)}")
    
    with col2:
        # Create page metadata
        page_metadata = {
            "page_number": page_data['page_number'],
            "detections": [det.to_dict() for det in detections],
            "ocr_regions": len(page_data['ocr_results']),
            "ocr_engine": page_data['engine_used']
        }
        
        metadata_bytes = json.dumps(page_metadata, indent=2).encode()
        
        st.download_button(
            label=f"📋 Download Page {page_data['page_number']} Metadata",
            data=metadata_bytes,
            file_name=f"page_{page_data['page_number']:03d}_metadata.json",
            mime="application/json",
            key=f"metadata_page_{page_data['page_number']}"
        )


def main():
    """Main Streamlit app."""
    setup_page_config()
    
    # Header
    st.title("🔒 Blurify - PII Redaction Tool")
    st.markdown("""
    **Privacy-first local redaction of personally identifiable information (PII) in documents.**
    
    Upload an image or PDF to automatically detect and redact sensitive information like emails, 
    phone numbers, names, dates, and government IDs.
    """)
    
    # Sidebar configuration
    config = create_sidebar_config()
    
    # Initialize components
    if 'components' not in st.session_state:
        with st.spinner("Initializing Blurify components..."):
            components = initialize_components(config)
            st.session_state.components = components
    else:
        components = st.session_state.components
    
    # Main content
    st.header("📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['jpg', 'jpeg', 'png', 'pdf', 'tiff', 'bmp'],
        help="Supported formats: JPG, PNG, PDF, TIFF, BMP"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 Uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Check file size
        max_size = 50 * 1024 * 1024  # 50MB limit
        if uploaded_file.size > max_size:
            st.error(f"File too large ({uploaded_file.size:,} bytes). Maximum size is {max_size:,} bytes (50MB).")
        else:
            # Only clear previous results if this is actually a different file
            current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if 'last_file_key' not in st.session_state:
                st.session_state.last_file_key = current_file_key
            
            # Clear results only if file changed
            if st.session_state.last_file_key != current_file_key:
                if 'results' in st.session_state:
                    del st.session_state.results
                st.session_state.last_file_key = current_file_key
            
            # Process button
            if st.button("🚀 Process File", type="primary"):
                
                # Clear any existing results
                if 'results' in st.session_state:
                    del st.session_state.results
                
                # Add progress container
                progress_container = st.container()
                
                with progress_container:
                    st.info("🔄 Processing file... This may take a few moments.")
                    
                    # Process the file
                    try:
                        result = process_uploaded_file(uploaded_file, components, config)
                        
                        if result is not None:
                            if result[0] == "multipage":
                                # Multi-page PDF results
                                _, all_pages_data, processing_time, summary_data = result
                                
                                # Store multi-page results
                                st.session_state.results = {
                                    'type': 'multipage',
                                    'all_pages_data': all_pages_data,
                                    'processing_time': processing_time,
                                    'summary_data': summary_data
                                }
                                
                            else:  # "singlepage"
                                # Single page results
                                _, original_image, redacted_image, processing_time, results_data = result
                                
                                # Store single-page results
                                st.session_state.results = {
                                    'type': 'singlepage',
                                    'original_image': original_image,
                                    'redacted_image': redacted_image,
                                    'processing_time': processing_time,
                                    'results_data': results_data
                                }
                            
                            st.success("✅ Processing completed successfully!")
                            
                        else:
                            st.error("❌ Processing failed. Please check the error messages above.")
                            
                    except Exception as e:
                        st.error(f"❌ Unexpected error during processing: {str(e)}")
                        import traceback
                        st.error(f"Debug info: {traceback.format_exc()}")
    
    # Display results if available
    if 'results' in st.session_state:
        st.divider()
        
        if st.session_state.results['type'] == 'multipage':
            # Display multi-page results
            display_multipage_results(
                st.session_state.results['all_pages_data'],
                st.session_state.results['processing_time'],
                st.session_state.results['summary_data']
            )
        else:  # singlepage
            # Display single-page results  
            display_results(
                st.session_state.results['original_image'],
                st.session_state.results['redacted_image'],
                st.session_state.results['processing_time'],
                st.session_state.results['results_data']
            )
    
    # Footer
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔒 **Privacy**: All processing happens locally. No data is sent to external servers.")
    
    with col2:
        st.info("⚡ **Performance**: Results depend on your hardware and the complexity of the document.")
    
    with col3:
        st.info("🛠️ **Open Source**: Blurify is open source and customizable for your needs.")
    
    # System info
    with st.expander("ℹ️ System Information"):
        if components[0] is not None:  # OCR manager
            available_engines = components[0].get_available_engines()
            st.write(f"**Available OCR Engines**: {', '.join(available_engines)}")
        
        st.write(f"**Configuration**:")
        st.json({
            "redaction_mode": config.redaction.default_mode.value,
            "enabled_pii_types": [t.value for t in config.detection.enabled_pii_types],
            "confidence_threshold": config.detection.confidence_threshold,
            "ocr_engine": config.ocr.primary_engine
        })


if __name__ == "__main__":
    # Setup logging
    setup_root_logger("INFO")
    
    # Run the app
    main()

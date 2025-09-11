"""
PDF utilities for converting PDFs to images and back.

Supports PyMuPDF (fitz) as primary engine with pdf2image as fallback.
"""

from typing import List, Union, Tuple, Optional, Dict, Any
from pathlib import Path
import tempfile
import io
import numpy as np
from PIL import Image
import json

from .config import BlurifyConfig
from .logger import LoggerMixin


class PDFConverter(LoggerMixin):
    """
    PDF to image converter with multiple backend support.
    """
    
    def __init__(self, config: Optional[BlurifyConfig] = None):
        """
        Initialize PDF converter.
        
        Args:
            config: Blurify configuration (optional)
        """
        self.config = config
        self.primary_engine = None
        self.fallback_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self) -> None:
        """Initialize available PDF engines."""
        engines_available = []
        
        # Try PyMuPDF (fitz) first
        try:
            import fitz  # PyMuPDF
            self.primary_engine = "pymupdf"
            engines_available.append("pymupdf")
            self.log_info("PyMuPDF (fitz) available as primary PDF engine")
        except ImportError:
            self.log_warning("PyMuPDF not available. Install with: pip install PyMuPDF")
        
        # Try pdf2image as fallback
        try:
            import pdf2image
            if self.primary_engine is None:
                self.primary_engine = "pdf2image"
            else:
                self.fallback_engine = "pdf2image"
            engines_available.append("pdf2image")
            self.log_info("pdf2image available")
        except ImportError:
            self.log_warning("pdf2image not available. Install with: pip install pdf2image")
        
        if not engines_available:
            raise RuntimeError(
                "No PDF engines available. Please install PyMuPDF or pdf2image:\n"
                "pip install PyMuPDF  # Recommended\n"
                "pip install pdf2image  # Requires poppler-utils system package"
            )
        
        self.log_info(f"PDF engines available: {engines_available}")
    
    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        dpi: int = 200,
        image_format: str = "PNG"
    ) -> List[Path]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save images (optional, uses temp if not provided)
            dpi: DPI for image conversion
            image_format: Output image format (PNG, JPEG, etc.)
            
        Returns:
            List of paths to converted images
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Use temporary directory if no output directory specified
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="blurify_pdf_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try primary engine first
        try:
            if self.primary_engine == "pymupdf":
                return self._pdf_to_images_pymupdf(pdf_path, output_dir, dpi, image_format)
            elif self.primary_engine == "pdf2image":
                return self._pdf_to_images_pdf2image(pdf_path, output_dir, dpi, image_format)
        except Exception as e:
            self.log_warning(f"Primary engine {self.primary_engine} failed: {e}")
        
        # Try fallback engine
        if self.fallback_engine:
            try:
                if self.fallback_engine == "pdf2image":
                    return self._pdf_to_images_pdf2image(pdf_path, output_dir, dpi, image_format)
                elif self.fallback_engine == "pymupdf":
                    return self._pdf_to_images_pymupdf(pdf_path, output_dir, dpi, image_format)
            except Exception as e:
                self.log_error(f"Fallback engine {self.fallback_engine} failed: {e}")
        
        raise RuntimeError("All PDF conversion engines failed")
    
    def _pdf_to_images_pymupdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        dpi: int,
        image_format: str
    ) -> List[Path]:
        """Convert PDF using PyMuPDF."""
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF not available")
        
        image_paths = []
        
        try:
            # Open PDF
            pdf_doc = fitz.open(str(pdf_path))
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                
                # Calculate zoom factor for desired DPI
                # PyMuPDF default is 72 DPI
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Save image
                output_path = output_dir / f"page_{page_num + 1:03d}.{image_format.lower()}"
                
                if image_format.upper() == "PNG":
                    pix.save(str(output_path))
                else:
                    # Convert to PIL Image for other formats
                    img_data = pix.tobytes("ppm")
                    pil_image = Image.open(io.BytesIO(img_data))
                    pil_image.save(str(output_path), format=image_format.upper())
                
                image_paths.append(output_path)
                self.log_debug(f"Converted page {page_num + 1} to {output_path}")
            
            pdf_doc.close()
            self.log_info(f"PyMuPDF: Converted {len(image_paths)} pages from {pdf_path}")
            
        except Exception as e:
            self.log_error(f"PyMuPDF conversion failed: {e}")
            raise
        
        return image_paths
    
    def _pdf_to_images_pdf2image(
        self,
        pdf_path: Path,
        output_dir: Path,
        dpi: int,
        image_format: str
    ) -> List[Path]:
        """Convert PDF using pdf2image."""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise RuntimeError("pdf2image not available")
        
        image_paths = []
        
        try:
            # Convert PDF to images
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt=image_format.lower()
            )
            
            for i, image in enumerate(images):
                output_path = output_dir / f"page_{i + 1:03d}.{image_format.lower()}"
                image.save(str(output_path), format=image_format.upper())
                image_paths.append(output_path)
                self.log_debug(f"Converted page {i + 1} to {output_path}")
            
            self.log_info(f"pdf2image: Converted {len(image_paths)} pages from {pdf_path}")
            
        except Exception as e:
            self.log_error(f"pdf2image conversion failed: {e}")
            # Check if poppler is installed
            if "poppler" in str(e).lower():
                self.log_error(
                    "pdf2image requires poppler-utils. Install with:\n"
                    "Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    "macOS: brew install poppler\n"
                    "Windows: Download from https://poppler.freedesktop.org/"
                )
            raise
        
        return image_paths
    
    def images_to_pdf(
        self,
        image_paths: List[Union[str, Path]],
        output_pdf_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Convert images back to PDF.
        
        Args:
            image_paths: List of image file paths
            output_pdf_path: Path for output PDF
            metadata: Optional metadata to embed
            
        Returns:
            Path to created PDF
        """
        if not image_paths:
            raise ValueError("No image paths provided")
        
        output_pdf_path = Path(output_pdf_path)
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try PyMuPDF first for better control
        if self.primary_engine == "pymupdf" or self.fallback_engine == "pymupdf":
            try:
                return self._images_to_pdf_pymupdf(image_paths, output_pdf_path, metadata)
            except Exception as e:
                self.log_warning(f"PyMuPDF PDF creation failed: {e}")
        
        # Fallback to PIL
        try:
            return self._images_to_pdf_pil(image_paths, output_pdf_path)
        except Exception as e:
            self.log_error(f"PIL PDF creation failed: {e}")
            raise
    
    def _images_to_pdf_pymupdf(
        self,
        image_paths: List[Union[str, Path]],
        output_pdf_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create PDF using PyMuPDF."""
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF not available")
        
        try:
            # Create new PDF document
            pdf_doc = fitz.open()
            
            for image_path in image_paths:
                image_path = Path(image_path)
                if not image_path.exists():
                    self.log_warning(f"Image not found: {image_path}")
                    continue
                
                # Open image and get dimensions
                pil_image = Image.open(image_path)
                img_width, img_height = pil_image.size
                
                # Create new page with image dimensions
                # Convert pixels to points (72 DPI)
                page_width = img_width * 72 / 200  # Assuming 200 DPI
                page_height = img_height * 72 / 200
                
                page = pdf_doc.new_page(width=page_width, height=page_height)
                
                # Insert image
                page.insert_image(
                    fitz.Rect(0, 0, page_width, page_height),
                    filename=str(image_path)
                )
            
            # Add metadata if provided
            if metadata:
                pdf_doc.set_metadata({
                    "title": "Blurify Redacted Document",
                    "author": "Blurify",
                    "subject": "PII Redacted Document",
                    "creator": "Blurify PII Redaction Tool",
                    "producer": "Blurify with PyMuPDF"
                })
            
            # Save PDF
            pdf_doc.save(str(output_pdf_path))
            pdf_doc.close()
            
            self.log_info(f"Created PDF with {len(image_paths)} pages: {output_pdf_path}")
            
        except Exception as e:
            self.log_error(f"PyMuPDF PDF creation failed: {e}")
            raise
        
        return output_pdf_path
    
    def _images_to_pdf_pil(
        self,
        image_paths: List[Union[str, Path]],
        output_pdf_path: Path
    ) -> Path:
        """Create PDF using PIL."""
        try:
            images = []
            
            for image_path in image_paths:
                image_path = Path(image_path)
                if not image_path.exists():
                    self.log_warning(f"Image not found: {image_path}")
                    continue
                
                # Load and convert image
                pil_image = Image.open(image_path)
                
                # Convert to RGB if needed (PDF requires RGB)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                images.append(pil_image)
            
            if not images:
                raise ValueError("No valid images found")
            
            # Save as PDF
            first_image = images[0]
            other_images = images[1:] if len(images) > 1 else []
            
            first_image.save(
                str(output_pdf_path),
                format="PDF",
                save_all=True,
                append_images=other_images
            )
            
            self.log_info(f"Created PDF with {len(images)} pages using PIL: {output_pdf_path}")
            
        except Exception as e:
            self.log_error(f"PIL PDF creation failed: {e}")
            raise
        
        return output_pdf_path
    
    def get_pdf_info(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try PyMuPDF first
        if self.primary_engine == "pymupdf" or self.fallback_engine == "pymupdf":
            try:
                return self._get_pdf_info_pymupdf(pdf_path)
            except Exception as e:
                self.log_warning(f"PyMuPDF PDF info failed: {e}")
        
        # Fallback to basic file info
        return {
            "page_count": 0,
            "file_size": pdf_path.stat().st_size,
            "metadata": {},
            "engine": "none"
        }
    
    def _get_pdf_info_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Get PDF info using PyMuPDF."""
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF not available")
        
        try:
            pdf_doc = fitz.open(str(pdf_path))
            
            info = {
                "page_count": pdf_doc.page_count,
                "file_size": pdf_path.stat().st_size,
                "metadata": pdf_doc.metadata,
                "engine": "pymupdf"
            }
            
            # Get first page dimensions
            if pdf_doc.page_count > 0:
                first_page = pdf_doc[0]
                rect = first_page.rect
                info["page_dimensions"] = {
                    "width": rect.width,
                    "height": rect.height
                }
            
            pdf_doc.close()
            return info
            
        except Exception as e:
            self.log_error(f"PyMuPDF PDF info extraction failed: {e}")
            raise


# Utility functions for common PDF operations
def split_pdf_by_pages(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    pages_per_split: int = 10
) -> List[Path]:
    """
    Split a large PDF into smaller PDFs.
    
    Args:
        pdf_path: Input PDF path
        output_dir: Output directory for split PDFs
        pages_per_split: Number of pages per split file
        
    Returns:
        List of paths to split PDF files
    """
    converter = PDFConverter()
    
    # Get PDF info
    pdf_info = converter.get_pdf_info(pdf_path)
    total_pages = pdf_info.get("page_count", 0)
    
    if total_pages == 0:
        raise ValueError("Could not determine PDF page count")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_files = []
    
    try:
        import fitz
        
        pdf_doc = fitz.open(str(pdf_path))
        
        for start_page in range(0, total_pages, pages_per_split):
            end_page = min(start_page + pages_per_split - 1, total_pages - 1)
            
            # Create new document with selected pages
            split_doc = fitz.open()
            split_doc.insert_pdf(pdf_doc, from_page=start_page, to_page=end_page)
            
            # Save split document
            split_filename = f"split_{start_page + 1:03d}_{end_page + 1:03d}.pdf"
            split_path = output_dir / split_filename
            split_doc.save(str(split_path))
            split_doc.close()
            
            split_files.append(split_path)
        
        pdf_doc.close()
        
    except ImportError:
        raise RuntimeError("PDF splitting requires PyMuPDF")
    except Exception as e:
        raise RuntimeError(f"PDF splitting failed: {e}")
    
    return split_files

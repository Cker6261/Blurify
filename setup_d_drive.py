"""
Configure EasyOCR and other ML libraries to use D: drive for model storage.
"""
import os
import tempfile
from pathlib import Path

def setup_d_drive_cache():
    """Set up D: drive directories for model caching."""
    
    # Create cache directories on D: drive
    d_cache_root = Path("D:/ml_cache")
    d_cache_root.mkdir(exist_ok=True)
    
    # EasyOCR cache
    easyocr_cache = d_cache_root / "easyocr"
    easyocr_cache.mkdir(exist_ok=True)
    
    # PyTorch cache
    torch_cache = d_cache_root / "torch"
    torch_cache.mkdir(exist_ok=True)
    
    # HuggingFace cache (if needed)
    hf_cache = d_cache_root / "huggingface"
    hf_cache.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ['EASYOCR_MODULE_PATH'] = str(easyocr_cache)
    os.environ['TORCH_HOME'] = str(torch_cache)
    os.environ['HF_HOME'] = str(hf_cache)
    os.environ['TRANSFORMERS_CACHE'] = str(hf_cache)
    
    # Set temporary directory to D: drive
    temp_dir = d_cache_root / "temp"
    temp_dir.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(temp_dir)
    os.environ['TEMP'] = str(temp_dir)
    os.environ['TMP'] = str(temp_dir)
    
    print(f"✅ Configured ML model cache directories:")
    print(f"   EasyOCR: {easyocr_cache}")
    print(f"   PyTorch: {torch_cache}")
    print(f"   Temp: {temp_dir}")
    
    return {
        'easyocr': easyocr_cache,
        'torch': torch_cache,
        'huggingface': hf_cache,
        'temp': temp_dir
    }

if __name__ == "__main__":
    setup_d_drive_cache()
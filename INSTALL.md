# 🚀 Blurify Installation Guide

## Quick Start (Windows)

**1. Prerequisites:**
- Python 3.10+ installed and added to PATH
- Git installed (to clone the repository)
- At least 2GB free space on D: drive (for model cache)

**2. Download and Setup:**
```batch
# Clone the repository
git clone https://github.com/your-username/blurify.git
cd blurify

# Run one-time setup (installs everything automatically)
setup_blurify.bat
```

**3. Run Blurify:**
```batch
# Start the web interface (opens in browser at http://localhost:8501)
run_blurify.bat

# OR test CLI functionality
test_blurify.bat
```

**That's it! 🎉**

## Manual Installation (Linux/macOS)

**1. Prerequisites:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils

# macOS
brew install tesseract poppler
```

**2. Python Setup:**
```bash
# Clone and setup
git clone https://github.com/your-username/blurify.git
cd blurify

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm
```

**3. Run:**
```bash
# Web interface
streamlit run streamlit_app.py

# CLI
python -m blurify.cli --input document.pdf --output results/
```

## Features

- **🔒 Privacy-First**: All processing happens locally
- **📄 Multi-Format**: Images (JPG, PNG, TIFF, BMP) and PDFs
- **🤖 Smart Detection**: Emails, phones, names, dates, Indian IDs
- **🎯 Redaction Modes**: Blur or mask sensitive information
- **📊 Web Interface**: Easy-to-use Streamlit UI
- **⚡ CLI Tool**: For batch processing and automation

## Troubleshooting

**Python not found:**
- Install Python 3.10+ from python.org
- Make sure it's added to PATH during installation

**Setup fails:**
- Ensure you have internet connection (downloads models)
- Check you have write access to D: drive
- Run as administrator if needed

**Need help?**
- Check the full README.md for detailed documentation
- Open an issue on GitHub for support
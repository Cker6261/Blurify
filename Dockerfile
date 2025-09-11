# Dockerfile for Blurify PII Redaction Tool
# Uses Ubuntu base image with system dependencies

FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and pip
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Poppler for PDF processing
    poppler-utils \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Other utilities
    git \
    wget \
    curl \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Install Blurify in development mode
RUN pip install -e .

# Generate demo data
RUN cd demo_data && python generate_demo_data.py

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative commands:
# Run CLI: docker run blurify python -m blurify.cli --help
# Run tests: docker run blurify python -m pytest tests/ -v
# Bash shell: docker run -it blurify bash

# Build instructions:
# docker build -t blurify .
# docker run -p 8501:8501 blurify

# For development with volume mount:
# docker run -p 8501:8501 -v $(pwd):/app blurify

# For CLI usage:
# docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output blurify \
#   python -m blurify.cli --input input/ --output output/

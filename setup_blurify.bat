@echo off
echo ================================================
echo       🔧 Blurify Setup & Installation
echo ================================================
echo.
echo This will set up Blurify with all dependencies...
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Installing core packages...
    pip install streamlit easyocr opencv-python pillow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install spacy pytesseract pymupdf pdf2image numpy scikit-image
    pip install --no-deps presidio-analyzer presidio-anonymizer
)

REM Download spaCy model
echo Downloading spaCy English model...
python -m spacy download en_core_web_sm

REM Install Blurify package
echo Installing Blurify package...
pip install -e .

REM Setup D: drive cache
echo Setting up D: drive cache...
python setup_d_drive.py

echo.
echo ================================================
echo ✅ Setup Complete!
echo ================================================
echo.
echo You can now run Blurify using: run_blurify.bat
echo.
pause
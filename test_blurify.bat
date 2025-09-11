@echo off
echo ================================================
echo        🧪 Blurify CLI Test & Demo
echo ================================================
echo.
echo Testing Blurify CLI functionality...
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_blurify.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Test CLI with demo data
echo Testing CLI with demo data...
echo.

if exist "demo_data" (
    echo Running CLI test on demo data...
    blurify demo_data --output cli_test_output --verbose
    
    echo.
    echo ✅ CLI test complete! Check cli_test_output folder for results.
) else (
    echo Creating test image...
    python -c "
from PIL import Image, ImageDraw, ImageFont
import os

# Create a simple test image
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)

# Add some test PII data
test_text = [
    'Name: John Doe',
    'Email: john.doe@example.com',
    'Phone: +1-555-123-4567',
    'Date: 2025-01-15',
    'SSN: 123-45-6789'
]

try:
    font = ImageFont.load_default()
except:
    font = None

y = 50
for text in test_text:
    draw.text((50, y), text, fill='black', font=font)
    y += 50

os.makedirs('test_images', exist_ok=True)
img.save('test_images/sample_pii.png')
print('✅ Test image created: test_images/sample_pii.png')
"
    
    echo Testing CLI with generated test image...
    blurify test_images/sample_pii.png --output cli_test_output --verbose
    
    echo.
    echo ✅ CLI test complete! Check cli_test_output folder for results.
)

echo.
echo Press any key to continue...
pause
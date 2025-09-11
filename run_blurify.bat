@echo off
echo ================================================
echo          🔒 Blurify PII Redaction Tool
echo ================================================
echo.
echo Starting Blurify web interface...
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_blurify.bat first to set up the environment.
    pause
    exit /b 1
)

REM Activate virtual environment and start Streamlit
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Streamlit server...
echo.
echo ✅ Blurify will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ================================================

streamlit run streamlit_app.py --server.port 8501 --server.address localhost

echo.
echo Blurify has been stopped.
pause
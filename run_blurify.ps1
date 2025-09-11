#!/usr/bin/env pwsh

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "          🔒 Blurify PII Redaction Tool" -ForegroundColor Green  
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Blurify web interface..." -ForegroundColor Yellow
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup_blurify.ps1 first to set up the environment." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Virtual environment found" -ForegroundColor Green

# Start Streamlit using the virtual environment
Write-Host "Starting Streamlit server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ Blurify will open in your browser at: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

try {
    & ".\.venv\Scripts\streamlit.exe" run streamlit_app.py --server.port 8501 --server.address localhost
} catch {
    Write-Host "Error starting Streamlit: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Blurify has been stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
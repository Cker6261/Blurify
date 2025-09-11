#!/usr/bin/env pwsh

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       🔒 Blurify Setup & Installation" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Setting up Blurify in: $ScriptDir" -ForegroundColor Yellow
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate and install packages
Write-Host "Installing Python packages..." -ForegroundColor Yellow
& ".\.venv\Scripts\pip.exe" install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install packages" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Python packages installed" -ForegroundColor Green

# Create D: drive cache directory
Write-Host "Setting up D: drive cache..." -ForegroundColor Yellow
$CacheDir = "D:\ml_cache"
if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
    Write-Host "✅ Created cache directory: $CacheDir" -ForegroundColor Green
} else {
    Write-Host "✅ Cache directory already exists: $CacheDir" -ForegroundColor Green
}

# Setup Streamlit config
Write-Host "Configuring Streamlit..." -ForegroundColor Yellow
$StreamlitDir = ".streamlit"
if (-not (Test-Path $StreamlitDir)) {
    New-Item -ItemType Directory -Path $StreamlitDir -Force | Out-Null
}

$ConfigContent = @"
[server]
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[theme]
primaryColor = "#ff6b6b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"@

$ConfigContent | Out-File -FilePath ".streamlit\config.toml" -Encoding UTF8
Write-Host "✅ Streamlit configured" -ForegroundColor Green

# Download spaCy model
Write-Host "Downloading spaCy model..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m spacy download en_core_web_sm

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🎉 Blurify Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: .\run_blurify.ps1 to start the web interface" -ForegroundColor White
Write-Host "2. Or run: .\test_blurify.ps1 to test CLI functionality" -ForegroundColor White
Write-Host ""
Write-Host "Cache location: D:\ml_cache" -ForegroundColor Cyan
Write-Host "Web interface: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
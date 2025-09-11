#!/usr/bin/env pwsh

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "          🔒 Blurify CLI Testing" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup_blurify.ps1 first to set up the environment." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Virtual environment found" -ForegroundColor Green
Write-Host ""

# Test with available files
Write-Host "Looking for test files..." -ForegroundColor Yellow

$TestFiles = @()
Get-ChildItem -Path . -Filter "*.pdf" | ForEach-Object { $TestFiles += $_.Name }
Get-ChildItem -Path . -Filter "*.docx" | ForEach-Object { $TestFiles += $_.Name }
Get-ChildItem -Path . -Filter "*.jpg" | ForEach-Object { $TestFiles += $_.Name }
Get-ChildItem -Path . -Filter "*.png" | ForEach-Object { $TestFiles += $_.Name }

if ($TestFiles.Count -eq 0) {
    Write-Host "No test files found in current directory." -ForegroundColor Yellow
    Write-Host "Please add some PDF, DOCX, JPG, or PNG files to test." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Found test files:" -ForegroundColor Green
for ($i = 0; $i -lt $TestFiles.Count; $i++) {
    Write-Host "  $($i + 1). $($TestFiles[$i])" -ForegroundColor White
}
Write-Host ""

# Let user choose file
Write-Host "Enter file number to test (1-$($TestFiles.Count)): " -NoNewline -ForegroundColor Yellow
$Choice = Read-Host

try {
    $FileIndex = [int]$Choice - 1
    if ($FileIndex -lt 0 -or $FileIndex -ge $TestFiles.Count) {
        throw "Invalid selection"
    }
    $TestFile = $TestFiles[$FileIndex]
} catch {
    Write-Host "❌ Invalid selection. Using first file." -ForegroundColor Red
    $TestFile = $TestFiles[0]
}

Write-Host ""
Write-Host "Testing CLI with file: $TestFile" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

# Run CLI test
$OutputDir = "test_output_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "Creating output directory: $OutputDir" -ForegroundColor Yellow

try {
    & ".\.venv\Scripts\python.exe" -m blurify.cli --input "$TestFile" --output "$OutputDir" --mode blur --confidence 0.7
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ CLI test completed successfully!" -ForegroundColor Green
        Write-Host "Check the output in: $OutputDir" -ForegroundColor Cyan
        
        # Show output files
        if (Test-Path $OutputDir) {
            $OutputFiles = Get-ChildItem -Path $OutputDir
            if ($OutputFiles.Count -gt 0) {
                Write-Host ""
                Write-Host "Generated files:" -ForegroundColor Green
                $OutputFiles | ForEach-Object { Write-Host "  • $($_.Name)" -ForegroundColor White }
            }
        }
    } else {
        Write-Host ""
        Write-Host "❌ CLI test failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error running CLI test: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "CLI test complete!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
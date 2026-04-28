<#
Robust virtual environment creator for Windows.

Usage:
  .\create_venv.ps1

This script will try:
  1) python -m venv .venv
  2) python -m ensurepip --upgrade
  3) download get-pip.py and install pip
  4) fallback: instruct user
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "Creating virtual environment .venv (PowerShell)..."
try {
    python -m venv .venv
    Write-Host "venv created successfully. Activate with: .\.venv\Scripts\Activate.ps1"
    exit 0
} catch {
    Write-Warning "python -m venv failed: attempting to ensure pip is available..."
}

try {
    python -m ensurepip --upgrade
    python -m venv .venv
    Write-Host "ensurepip succeeded and venv created. Activate with: .\.venv\Scripts\Activate.ps1"
    exit 0
} catch {
    Write-Warning "ensurepip or venv creation still failed; will attempt get-pip.py install."
}

$getpip = "$env:TEMP\get-pip.py"
try {
    Write-Host "Downloading get-pip.py to $getpip"
    Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile $getpip -UseBasicParsing -ErrorAction Stop
    Write-Host "Installing pip via get-pip.py..."
    python $getpip
    Write-Host "Retrying venv creation..."
    python -m venv .venv
    Write-Host "venv created successfully after installing pip. Activate with: .\.venv\Scripts\Activate.ps1"
    Remove-Item -Force $getpip -ErrorAction SilentlyContinue
    exit 0
} catch {
    Write-Error "Automatic venv creation failed. Last error: $($_.Exception.Message)"
    Write-Host "Manual recovery options:" -ForegroundColor Yellow
    Write-Host "1) Ensure your Python installation includes the 'ensurepip' module or reinstall Python from https://www.python.org/downloads/ (select 'Install pip')."
    Write-Host "2) Run as administrator if permission issues are suspected."
    Write-Host "3) As a last resort, install virtualenv and create a venv manually:"
    Write-Host "   python -m pip install --user virtualenv" -ForegroundColor Cyan
    Write-Host "   python -m virtualenv .venv" -ForegroundColor Cyan
    exit 1
}

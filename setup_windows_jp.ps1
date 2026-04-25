# Style-Bert-VITS2 Windows Setup Script
# Supports Japanese username environments (e.g. C:\Users\川上貴大\...)
# CPU-only mode (no NVIDIA GPU required)
#
# Usage:
#   PowerShell -ExecutionPolicy Bypass -File setup_windows_jp.ps1
#
# Options:
#   -SkipModels    Skip downloading default voice models (jvnv-F1-jp etc.)
#   -SkipVC        Skip Visual C++ Redistributable check/install

param(
    [switch]$SkipModels,
    [switch]$SkipVC
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Step { param([string]$msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-OK   { param([string]$msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$msg) Write-Host "[FAIL] $msg" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# Step 1: Visual C++ 2015-2022 Redistributable
# ---------------------------------------------------------------------------
Write-Step "Visual C++ 2015-2022 Redistributable"

if (-not $SkipVC) {
    $vcKey = "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    $vcInstalled = (Test-Path $vcKey) -and ((Get-ItemProperty $vcKey -ErrorAction SilentlyContinue).Installed -eq 1)

    if ($vcInstalled) {
        Write-OK "Already installed"
    } else {
        Write-Host "Installing Visual C++ Redistributable..."
        winget install Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "winget install failed (exit $LASTEXITCODE). Continuing anyway."
        } else {
            Write-OK "Installed"
        }
    }
} else {
    Write-Warn "Skipped (--SkipVC)"
}

# ---------------------------------------------------------------------------
# Step 2: Python 3.10 at ASCII path (C:\Python310)
# ---------------------------------------------------------------------------
Write-Step "Python 3.10 at C:\Python310"

$Python310 = "C:\Python310\python.exe"

if (Test-Path $Python310) {
    $ver = & $Python310 --version 2>&1
    Write-OK "$ver found at $Python310"
} else {
    Write-Host "Python 3.10 not found at C:\Python310. Installing via winget..."
    winget install Python.Python.3.10 --location C:\Python310 --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python310)) {
        Write-Fail "Failed to install Python 3.10. Please install manually to C:\Python310"
        exit 1
    }
    Write-OK "Python 3.10 installed"
}

# ---------------------------------------------------------------------------
# Step 3: uv package manager
# ---------------------------------------------------------------------------
Write-Step "uv package manager"

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    Write-OK "uv $(uv --version) already installed"
} else {
    Write-Host "Installing uv..."
    & $Python310 -m pip install uv --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "pip install uv failed"; exit 1 }
    Write-OK "uv installed"
}

# ---------------------------------------------------------------------------
# Step 4: Virtual environment
# ---------------------------------------------------------------------------
Write-Step "Virtual environment (venv)"

$VenvPython = ".\venv\Scripts\python.exe"
$PyvenvCfg  = ".\venv\pyvenv.cfg"

$needRebuild = $false
if (Test-Path $PyvenvCfg) {
    $home = (Get-Content $PyvenvCfg | Select-String "^home").ToString()
    if ($home -notmatch "C:\\Python310") {
        Write-Warn "Existing venv points to wrong Python ($home). Rebuilding..."
        Remove-Item -Recurse -Force .\venv
        $needRebuild = $true
    } else {
        Write-OK "Existing venv OK"
    }
} else {
    $needRebuild = $true
}

if ($needRebuild) {
    Write-Host "Creating venv with C:\Python310..."
    & $Python310 -m venv venv
    if ($LASTEXITCODE -ne 0) { Write-Fail "venv creation failed"; exit 1 }
    Write-OK "venv created"
}

# ---------------------------------------------------------------------------
# Step 5: CPU PyTorch
# ---------------------------------------------------------------------------
Write-Step "PyTorch (CPU)"

$torchOk = & $VenvPython -c "import torch; print(torch.__version__)" 2>&1
if ($torchOk -match "2\.[34]") {
    Write-OK "torch $torchOk already installed"
} else {
    Write-Host "Installing torch 2.3.1+cpu and torchaudio..."
    uv pip install --python $VenvPython `
        "torch==2.3.1" "torchaudio==2.3.1" `
        --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Write-Fail "PyTorch install failed"; exit 1 }
    Write-OK "torch 2.3.1+cpu installed"
}

# ---------------------------------------------------------------------------
# Step 6: Requirements (with Windows/Japanese-env workarounds)
# ---------------------------------------------------------------------------
Write-Step "Python dependencies"

# Build a filtered requirements list: exclude torch/torchaudio (already installed)
# and faster-whisper (pinned version fails to build 'av' on Windows Python 3.10)
$reqs = Get-Content requirements.txt |
    Where-Object { $_ -notmatch '^\s*(torch|torchaudio|faster-whisper)' }
$reqs | Out-File -FilePath requirements-custom.txt -Encoding utf8

Write-Host "Installing requirements (excluding torch/torchaudio/faster-whisper)..."
uv pip install --python $VenvPython -r requirements-custom.txt
if ($LASTEXITCODE -ne 0) { Write-Fail "requirements install failed"; exit 1 }

# faster-whisper: use latest version (has prebuilt av wheels for Windows)
Write-Host "Installing faster-whisper (latest)..."
uv pip install --python $VenvPython faster-whisper
if ($LASTEXITCODE -ne 0) { Write-Fail "faster-whisper install failed"; exit 1 }

# soxr: required by transformers 4.57+ but not listed in requirements
Write-Host "Installing soxr..."
uv pip install --python $VenvPython soxr
if ($LASTEXITCODE -ne 0) { Write-Warn "soxr install failed (non-critical)" }

# transformers: pin to 4.x to stay compatible with torch 2.3.1
Write-Host "Pinning transformers to 4.x (compatible with torch 2.3.1)..."
uv pip install --python $VenvPython "transformers>=4.40,<5.0"
if ($LASTEXITCODE -ne 0) { Write-Fail "transformers install failed"; exit 1 }

Write-OK "All dependencies installed"

# ---------------------------------------------------------------------------
# Step 7: Download models
# ---------------------------------------------------------------------------
Write-Step "Download BERT / pretrained models"

$initArgs = @()
if ($SkipModels) {
    $initArgs += "--skip_default_models"
    Write-Warn "Skipping default voice models (--SkipModels)"
}

& $VenvPython initialize.py @initArgs
if ($LASTEXITCODE -ne 0) { Write-Fail "initialize.py failed"; exit 1 }
Write-OK "Models downloaded"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the Web UI, run:"
Write-Host "  .\venv\Scripts\python.exe app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then open http://127.0.0.1:7860 in your browser."
Write-Host ""

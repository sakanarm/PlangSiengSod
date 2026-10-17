<#
.SYNOPSIS
  Set up the Real-Time Voice Changer in .venv (no administrator rights needed).

.DESCRIPTION
  Default (lite, ~260 MB): robot voice, pass-through, VB-CABLE routing, diagnostics.
  -WithAI     : also install the optional AI (RVC) add-on (~2 GB, CPU PyTorch - no NVIDIA/CUDA)
                and download the official HuBERT base model (181 MB, SHA-256 verified).
  -WithRMVPE  : with -WithAI, also download the RMVPE pitch model (345 MB, optional).
  -Dev        : install pytest for running the unit tests.
  -Recreate   : delete and rebuild .venv (never done without this switch).

  This script never installs drivers. VB-CABLE must be installed by you (see docs\INSTALL_WINDOWS.md).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup.ps1
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Dev
#>
param(
    [switch]$WithAI,
    [switch]$WithRMVPE,
    [switch]$Dev,
    [switch]$Recreate
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

Step "Checking Python 3.12 (64-bit)"
$launcher = Get-Command py -ErrorAction SilentlyContinue
if (-not $launcher) {
    Write-Host "The Python launcher 'py' was not found. Install Python 3.12 x64 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH' and 'py launcher'), then run this script again." -ForegroundColor Red
    exit 1
}
& py -3.12 -c "import struct,sys; assert struct.calcsize('P')==8, '64-bit Python required'; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 x64 is required (py -3.12). Install it from python.org and run again." -ForegroundColor Red
    exit 1
}

if ((Test-Path $venv) -and $Recreate) {
    Step "Removing existing .venv (-Recreate)"
    Remove-Item -Recurse -Force -LiteralPath $venv
}
if (-not (Test-Path $py)) {
    Step "Creating virtual environment .venv"
    & py -3.12 -m venv $venv
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host "Using existing .venv (pass -Recreate to rebuild it)."
}

& $py -m pip install --disable-pip-version-check -q --upgrade pip
$req = "requirements.txt"
if ($WithAI) { $req = "requirements-ai.txt" }
Step "Installing pinned packages from $req"
& $py -m pip install --disable-pip-version-check -r $req
if ($LASTEXITCODE -ne 0) { Write-Host "Package installation failed - see the output above." -ForegroundColor Red; exit 1 }
if ($Dev) {
    & $py -m pip install --disable-pip-version-check -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

if ($WithAI) {
    Step "Downloading official base models (huggingface.co/lj1995/VoiceConversionWebUI, SHA-256 verified)"
    $dlArgs = @("-m", "rtvc.voice_conversion.assets", "--download")
    if ($WithRMVPE) { $dlArgs += "--with-rmvpe" }
    & $py @dlArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "Base model download failed; run setup again to retry." -ForegroundColor Red; exit 1 }
}

Step "Checking VB-CABLE"
& $py -c "from rtvc.audio import devices as d; import sys; sys.exit(0 if d.cable_installed(d.list_devices()) else 3)"
if ($LASTEXITCODE -eq 3) {
    Write-Host "VB-CABLE is NOT installed. Install it yourself (needs administrator rights + reboot):" -ForegroundColor Yellow
    Write-Host "  https://vb-audio.com/Cable/  ->  run VBCABLE_Setup_x64.exe as administrator -> reboot." -ForegroundColor Yellow
    Write-Host "  Details: docs\INSTALL_WINDOWS.md" -ForegroundColor Yellow
} elseif ($LASTEXITCODE -eq 0) {
    Write-Host "VB-CABLE found." -ForegroundColor Green
} else {
    Write-Host "Could not query audio devices (see error above)." -ForegroundColor Yellow
}

Step "Done"
Write-Host "Start the app with:  powershell -ExecutionPolicy Bypass -File .\run.ps1"

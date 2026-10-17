<#
.SYNOPSIS
  Start Parroty (real-time voice changer).
  -Diagnostics : print the hardware/audio report instead of opening the window.
  -Test        : run the unit tests (needs setup.ps1 -Dev).
  -CableTest   : verify the VB-CABLE path with a test tone (+ 5 s robot-voice engine run).
#>
param([switch]$Diagnostics, [switch]$Test, [switch]$CableTest)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Not set up yet. Run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Red
    exit 1
}
if ($Diagnostics) { & $py -m rtvc.hardware.diagnostics --verify-gpu; exit $LASTEXITCODE }
if ($Test) { & $py -m pytest -q tests; exit $LASTEXITCODE }
if ($CableTest) { & $py scripts\cable_loopback_test.py; exit $LASTEXITCODE }
& $py -m rtvc
exit $LASTEXITCODE

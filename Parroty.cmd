@echo off
rem Double-click to start Parroty (no PowerShell needed). Runs from this folder, no console window stays open.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Parroty is not set up yet. Run this once in PowerShell, in this folder:
    echo.
    echo     powershell -ExecutionPolicy Bypass -File .\setup.ps1
    echo.
    echo Details: docs\INSTALL_WINDOWS.md
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m rtvc

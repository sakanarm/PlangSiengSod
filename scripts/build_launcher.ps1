<#
.SYNOPSIS
  Build Parroty.exe (a ~100 KB launcher with the Parroty icon) in the project folder.

.DESCRIPTION
  Compiles scripts\launcher\Parroty.cs with the C# compiler that ships with Windows (.NET Framework 4,
  %WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe) - nothing is downloaded and no administrator rights are needed.
  The exe only starts .venv\Scripts\pythonw.exe -m rtvc, so setup.ps1 must have been run.
  The built exe is not committed (.gitignore). It looks for .venv next to itself (no fixed paths).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\build_launcher.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$src = Join-Path $PSScriptRoot "launcher\Parroty.cs"
$icon = Join-Path $root "rtvc\ui\assets\parroty.ico"
$exe = Join-Path $root "Parroty.exe"

if (-not (Test-Path -LiteralPath $csc)) {
    Write-Host "The .NET Framework 4 C# compiler was not found ($csc). Use Parroty.cmd instead." -ForegroundColor Yellow
    exit 2
}
& $csc /nologo /target:winexe /optimize+ /platform:anycpu "/win32icon:$icon" "/out:$exe" `
    /reference:System.Windows.Forms.dll $src
if ($LASTEXITCODE -ne 0) { Write-Host "Building Parroty.exe failed - see the output above." -ForegroundColor Red; exit 1 }
Write-Host "Built $exe" -ForegroundColor Green

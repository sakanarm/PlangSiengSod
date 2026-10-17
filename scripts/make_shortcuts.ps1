<#
.SYNOPSIS
  Create "Parroty" shortcuts (Start menu + Desktop) with the Parroty icon - for the current user, no administrator rights.

.DESCRIPTION
  The shortcuts open Parroty.exe (built by build_launcher.ps1 if missing), which starts .venv\Scripts\pythonw.exe -m rtvc
  in the project folder (no console window). They carry the app's taskbar ID, so "Pin to taskbar" / "Pin to Start" keep the Parroty icon.
  An existing Parroty.lnk that points somewhere else is left untouched.
  -NoDesktop : only the Start menu shortcut.
  -Remove    : delete the shortcuts this script made.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\make_shortcuts.ps1
#>
param([switch]$NoDesktop, [switch]$Remove)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
$exe = Join-Path $root "Parroty.exe"
$icon = Join-Path $root "rtvc\ui\assets\parroty.ico"
$appId = "Parroty.VoiceChanger"    # same ID the window sets (rtvc/ui/app.py)

$links = @(Join-Path ([Environment]::GetFolderPath("Programs")) "Parroty.lnk")
if (-not $NoDesktop) { $links += Join-Path ([Environment]::GetFolderPath("Desktop")) "Parroty.lnk" }
$shell = New-Object -ComObject WScript.Shell

function Ours($lnk) { (Test-Path -LiteralPath $lnk) -and ($shell.CreateShortcut($lnk).TargetPath -in @($exe, $pyw)) }

if ($Remove) {
    foreach ($lnk in $links) {
        if (Ours $lnk) { Remove-Item -LiteralPath $lnk; Write-Host "Removed $lnk" }
    }
    exit 0
}
if (-not (Test-Path -LiteralPath $pyw)) {
    Write-Host "Not set up yet. Run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Red
    exit 1
}
$target, $arguments = $exe, ""
if (-not (Test-Path -LiteralPath $exe)) {
    & (Join-Path $PSScriptRoot "build_launcher.ps1")
    if ($LASTEXITCODE -ne 0) { $target, $arguments = $pyw, "-m rtvc" }    # no compiler: start Python directly
}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ParrotyLink {
    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct PropertyKey { public Guid fmtid; public uint pid; }
    [StructLayout(LayoutKind.Explicit)]
    struct PropVariant { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; [FieldOffset(16)] public IntPtr pad; }
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName(string path, IntPtr bind, int flags, ref Guid iid,
        [MarshalAs(UnmanagedType.Interface)] out IPropertyStore store);

    public static void SetAppId(string lnk, string appId) {
        Guid iid = typeof(IPropertyStore).GUID;
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(lnk, IntPtr.Zero, 2 /* GPS_READWRITE */, ref iid, out store);
        try {
            PropertyKey key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };
            PropVariant v = new PropVariant { vt = 31 /* VT_LPWSTR */, p = Marshal.StringToCoTaskMemUni(appId) };
            try { store.SetValue(ref key, ref v); store.Commit(); }
            finally { Marshal.FreeCoTaskMem(v.p); }
        } finally { Marshal.ReleaseComObject(store); }
    }
}
"@

foreach ($lnk in $links) {
    if ((Test-Path -LiteralPath $lnk) -and -not (Ours $lnk)) {
        Write-Host "Skipped $lnk - a different shortcut with that name already exists." -ForegroundColor Yellow
        continue
    }
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = $target
    $s.Arguments = $arguments
    $s.WorkingDirectory = $root
    $s.IconLocation = "$icon,0"
    $s.Description = "Parroty - real-time voice changer"
    $s.Save()
    try { [ParrotyLink]::SetAppId($lnk, $appId) }
    catch { Write-Host "  (taskbar ID not set: $($_.Exception.Message) - the shortcut still works)" -ForegroundColor Yellow }
    Write-Host "Created $lnk" -ForegroundColor Green
}

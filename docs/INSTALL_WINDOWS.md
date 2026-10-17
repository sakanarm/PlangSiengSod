# Installation (Windows 10/11)

## 1. Python 3.12 (64-bit)
Install from https://www.python.org/downloads/windows/ – tick **py launcher** and
**Add python.exe to PATH**. Check: `py -3.12 --version`.

## 2. VB-CABLE (virtual audio cable) – you install this yourself
The app never installs drivers. Driver installation needs administrator rights:

1. Download the official package from https://vb-audio.com/Cable/ (VB-Audio Software).
2. Extract the ZIP, right-click **VBCABLE_Setup_x64.exe** → **Run as administrator** → *Install Driver*.
3. **Reboot.**
4. Check in *Settings → System → Sound*: playback device **CABLE Input**, recording device **CABLE Output**.

Recommended: in *Sound Control Panel* set both CABLE devices and your mic to
**48000 Hz** (*Properties → Advanced*).

## 3. Microphone permission
*Settings → Privacy & security → Microphone*: turn on **Microphone access** and
**Let desktop apps access your microphone**.

## 4. App setup
```powershell
cd <folder of this project>
powershell -ExecutionPolicy Bypass -File .\setup.ps1          # lite (~260 MB)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -CableTest # optional check
powershell -ExecutionPolicy Bypass -File .\run.ps1
```
`setup.ps1` reuses an existing `.venv`; it only deletes it with `-Recreate`.

Optional AI add-on (not needed for the robot voice): `.\setup.ps1 -WithAI` (~2 GB + 181 MB).

## Uninstall
Delete the project folder. VB-CABLE: run its setup as administrator → *Remove Driver* → reboot.

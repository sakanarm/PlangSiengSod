# Real-Time Voice Changer for Windows (Teams via VB-CABLE)

Microphone → real-time processing → **VB-CABLE** → Microsoft Teams.
Runs on a normal Windows 10/11 laptop (Intel CPU/graphics). No NVIDIA GPU, no CUDA.

```
[Physical mic] -> [this app: Robot voice / Pass-through] -> "CABLE Input" ==VB-CABLE==> "CABLE Output" -> [Teams microphone]
```

## Status (v0.1 – "lite")

| Feature | Status |
|---|---|
| Robot voice v1 (DSP effect, **not AI**) | ✅ working, tested on real devices |
| Pass-through (routing test) | ✅ |
| VB-CABLE detection, feedback-loop protection, test beep | ✅ |
| Level meters, measured latency / drops / underruns / errors, CPU & RAM | ✅ |
| Device-disconnect detection, error state, logs (`logs\rtvc.log`) | ✅ |
| Hardware diagnostics (Windows, CPU, RAM, GPU + driver, VB-CABLE) | ✅ |
| AI voice conversion (RVC) | 🚧 optional add-on, engine code in progress – not enabled in the UI |

The robot voice is a classic signal-processing effect (STFT phase zeroing + optional ring
modulation). It does not imitate any person and needs no model or download.

## Quick start

```powershell
# 1. one-time setup (lite, about 260 MB, no admin rights)
powershell -ExecutionPolicy Bypass -File .\setup.ps1
# 2. start
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

In the app: Microphone = your mic, Output → Teams = **CABLE Input (VB-Audio Virtual Cable)**,
press **Start**. In Teams choose microphone **CABLE Output (VB-Audio Virtual Cable)**.
Full steps: [docs/TEAMS_SETUP.md](docs/TEAMS_SETUP.md).

**Recording:** while running, press **● Record** and choose where to save. The processed voice
(exactly what goes to the output) is written as a 16-bit mono WAV. The default folder is `recordings/`.
Press **■ Stop recording** or **Stop** to finish. Files stay on this PC and are never uploaded.

### เริ่มใช้งานแบบย่อ (ภาษาไทย)
1. ติดตั้ง VB-CABLE เอง (ต้องใช้สิทธิ์ admin + รีบูต) – ดู [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md)
2. `setup.ps1` แล้ว `run.ps1`
3. ในแอป: Microphone = ไมค์จริง, Output → Teams = **CABLE Input** → กด **Start**
4. ใน Teams: Settings → Devices → Microphone = **CABLE Output**
5. แจ้งผู้ร่วมประชุมว่ากำลังใช้เอฟเฟกต์เสียง
6. อยากได้ไฟล์เสียง: ระหว่างที่รันอยู่ กด **● Record** → เลือกที่เก็บไฟล์ → กด **■ Stop recording** ระบบจะได้ไฟล์ .wav ของเสียงที่แปลงแล้ว (ค่าเริ่มต้นอยู่ในโฟลเดอร์ `recordings/`)

## Other commands

```powershell
.\run.ps1 -Diagnostics     # hardware / audio report
.\run.ps1 -CableTest       # test tone through VB-CABLE + 5 s robot-voice engine run
.\setup.ps1 -Dev; .\run.ps1 -Test   # unit tests
```

## Disk usage

| Install | Size |
|---|---|
| Lite (default) | ~260 MB (`.venv`) |
| `-WithAI` | +~2 GB (CPU PyTorch, ONNX Runtime DirectML) + 181 MB HuBERT base model |
| `-WithAI -WithRMVPE` | + 345 MB RMVPE pitch model (optional) |

## Responsible use
Use only your own voice or voices/models you are authorized to use. Do not use this to
impersonate someone, bypass identity checks or mislead meeting participants about who is
speaking. Recordings and models stay on this PC; nothing is uploaded.

## Project layout
`rtvc/audio` (devices, engine, processors) · `rtvc/hardware` (diagnostics) · `rtvc/ui` (tkinter app) ·
`rtvc/voice_conversion` (RVC add-on, in progress; vendored upstream code in `rvc_lib/`, MIT) ·
`tests/` · `scripts/` · `docs/`

Docs: **[User manual (Thai, HTML)](docs/MANUAL.html)**, also opened by the app's *Manual / คู่มือ* button · [INSTALL_WINDOWS](docs/INSTALL_WINDOWS.md) · [TEAMS_SETUP](docs/TEAMS_SETUP.md) ·
[TROUBLESHOOTING](docs/TROUBLESHOOTING.md) · [MODEL_TRAINING](docs/MODEL_TRAINING.md)

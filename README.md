# Parroty – real-time voice changer for Windows (Teams via VB-CABLE)

Microphone → real-time processing → **VB-CABLE** → Microsoft Teams.
Runs on a normal Windows 10/11 laptop (Intel CPU/graphics). No NVIDIA GPU, no CUDA.

```
[Physical mic] -> [this app: Robot / Other voices / Pass-through] -> "CABLE Input" ==VB-CABLE==> "CABLE Output" -> [Teams microphone]
```

## Status (v0.2 – "lite")

| Feature | Status |
|---|---|
| Robot voice v1 (DSP effect, **not AI**) | ✅ working, tested on real devices |
| Other voices: 13 person-like voices (incl. 4 "unwell": cold, sore throat, hoarse, hoarse own voice) + 12 effects (DSP, **not AI**), live switching | ✅ working, tested on real devices |
| Voice from sample (v0.2): measure pitch / formants / tone of a 10–20 s WAV/WebM/recording, save as an adjustable voice | ✅ unit + UI tests; please verify by ear |
| Pass-through (routing test) | ✅ |
| VB-CABLE detection, feedback-loop protection, test beep | ✅ |
| Level meters, measured latency / drops / underruns / delay fixes / errors, CPU & RAM | ✅ |
| "Test my voice": record 5 s, hear it back with the selected voice (headphones/speakers, never the cable), replay with another voice | ✅ unit tests + real-device UI check |
| Long calls: extra delay after a hiccup (or from mic/cable clock drift) is trimmed automatically | ✅ simulated-clock tests + real-device soak with injected stalls |
| Double-click launcher (`Parroty.exe` / `Parroty.cmd`) + Desktop/Start menu shortcuts | ✅ |
| Device-disconnect detection, error state, logs (`logs\rtvc.log`) | ✅ |
| Hardware diagnostics (Windows, CPU, RAM, GPU + driver, VB-CABLE) | ✅ |
| AI voice conversion (RVC) | 🚧 optional add-on, engine code in progress – not enabled in the UI |

The robot voice is a classic signal-processing effect (STFT phase zeroing + optional ring
modulation). It does not imitate any person and needs no model or download.

**Other voices** (`rtvc/audio/voices.py`) are DSP too: a phase-vocoder pitch shift with a separate
formant (spectral-envelope) shift, plus simple effects. Pitch / formant can be fine-tuned live.

| Voice | Pitch (st) | Formant (st) | Notes |
|---|---|---|---|
| Deep male (default) / Big man / Young male | −4 / −6 / +3 | −2 / −3.5 / +1 | |
| Female / Soft female / Child | +8 / +6 / +11 | +3 / +2.5 / +4.5 | |
| Elderly / Tiny / Gentle giant | +2 / 0 / 0 | +0.5 / +5 / −5 | Elderly adds a slow pitch tremor |
| Blocked nose (cold) | −1 | −0.5 | nasal/boxy EQ, top cut at 5 kHz |
| Sore throat / Hoarse | −1.5 / −2 | 0 / −0.5 | pitch jitter + breath layer + roughness (Hoarse adds light drive) |
| Hoarse, own voice | – | – | your unshifted voice + breath + roughness; ~21 ms added |
| Chipmunk / Monster | +12 / −7 | +8 / −5 | Monster adds saturation |
| Radio / Cave | 0 / 0 | 0 / 0 | band-pass + drive / convolution reverb |
| Whisper | – | – | pitch removed entirely, breathy noise + light echo |
| Ghost | – | – | whisper + slow flanger swirl + 3.5 s reverb |
| Alien | +3 | +3 | vibrato + ring modulation |
| Masked villain | −5 | −3 | muffled + metallic edge |
| Retro game (8-bit) | – | – | sample-and-hold to 6 kHz + 8-bit quantisation; **0 ms** added |
| Jet flanger | – | – | swept 1–5 ms delay with feedback; **0 ms** added |
| Underwater | – | – | 2.5 ms delay wobble at 1.2 Hz + 900 Hz low-pass; ~8 ms added |
| Stadium announcer | – | – | 300 Hz–4.5 kHz band + drive + 220 ms echo; **0 ms** added |

Limits: these effects change pitch and timbre but keep your rhythm, accent and intonation, so
people who know you may still recognise you. Changing an accent is not possible in real time (it needs
speech recognition + re-synthesis, i.e. seconds of delay). Algorithmic latency is ~43 ms (Radio/Cave/Masked villain/My voices ~64 ms;
Retro/Jet/Stadium 0 ms, Underwater ~8 ms, Hoarse own voice ~21 ms),
plus buffering and device latency.

**Voice from sample** (`rtvc/audio/voice_match.py`, page *Voice from sample*): load a 10–20 s
reference (WAV, WebM, MP3, M4A, OGG, FLAC, Opus) or record one, and record your own voice. The app
measures median pitch (YIN), formants F1–F3 (LPC) and the long-term spectrum (1/3-octave bands),
derives a pitch shift (±12 st), a formant shift (±8 st) and a 20-band tone EQ (100 Hz–8 kHz,
+6 / −10 dB), then renders your recording through those settings and **measures it again** – the
"After matching" column is that real measurement. The result is saved under *My voices* as numbers
only (`custom_voices\*.json`, git-ignored; no audio is stored) and runs in the same real-time
engine, so it works in Teams/Zoom through CABLE Input; the sliders stay adjustable. It gets you
*closer* to the reference, not the same person (rhythm, accent and phrasing stay yours). Use only
voices you are authorised to use, and never to impersonate someone.

## Quick start

```powershell
# 1. one-time setup (lite, about 260 MB, no admin rights)
powershell -ExecutionPolicy Bypass -File .\setup.ps1
# 2. start
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

After setup you can also start it by double-clicking **`Parroty.cmd`**, or build a small launcher with the
Parroty icon (`scripts\build_launcher.ps1` → `Parroty.exe`) and add Desktop/Start menu shortcuts
(`scripts\make_shortcuts.ps1`, current user, no admin rights). Both run `.venv\Scripts\pythonw.exe -m rtvc`
from the project folder without a console window; `run.ps1` still works and shows errors in the console.

**Hear what others will hear:** on the Voice page, *Test my voice* → **● Speak 5 s** records 5 s from your
microphone and plays it back with the selected voice on the Monitor device (or the Windows default output,
never VB-CABLE); **▶ Play again** replays the same take with whatever voice/settings are selected now. The take
stays in memory only and the button works only while stopped. For a live check, set *Monitor (headphones)* to
your headphones and press Start (use headphones, not speakers, or you get echo), or press **● Record** and play
the WAV back, or use Teams *Settings → Devices → Make a test call* with microphone = CABLE Output.

**Long calls:** if the PC stalls for a moment (updates, heavy apps), the extra audio it leaves queued is
removed automatically within a few seconds, cut in your speech pauses (or at most 10 ms per block after 8 s
of non-stop speech); the same mechanism absorbs mic/VB-CABLE clock drift. The stats line shows the count as
*Delay fixes*. Only your microphone is processed – other participants' voices are not touched, as long as
the Teams *Speaker* is your real headphones/speakers and **not** CABLE Input.

In the app: Microphone = your mic, Output → Teams = **CABLE Input (VB-Audio Virtual Cable)**,
press **Start**. In Teams choose microphone **CABLE Output (VB-Audio Virtual Cable)**.
Full steps: [docs/TEAMS_SETUP.md](docs/TEAMS_SETUP.md).

**Recording:** while running, press **● Record** and choose where to save. The processed voice
(exactly what goes to the output) is written as a 16-bit mono WAV. The default folder is `recordings/`.
Press **■ Stop recording** or **Stop** to finish. Files stay on this PC and are never uploaded.

### เริ่มใช้งานแบบย่อ (ภาษาไทย)
1. ติดตั้ง VB-CABLE เอง (ต้องใช้สิทธิ์ admin + รีบูต) – ดู [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md)
2. `setup.ps1` แล้ว `run.ps1`
3. ในแอป: หน้า **Audio devices** เลือก Microphone = ไมค์จริง แล้วกด **Use VB-CABLE** → หน้า **Voice** คลิกเลือกเสียง → กด **▶ Start** (เปลี่ยนเสียงระหว่างทำงานได้เลย)
4. ใน Teams: Settings → Devices → Microphone = **CABLE Output** (Zoom / Meet / Discord / LINE ก็เลือกไมค์ = **CABLE Output** แบบเดียวกัน)
5. แจ้งผู้ร่วมประชุมว่ากำลังใช้เอฟเฟกต์เสียง
6. อยากได้ไฟล์เสียง: ระหว่างที่รันอยู่ กด **● Record** → เลือกที่เก็บไฟล์ → กด **■ Stop recording** ระบบจะได้ไฟล์ .wav ของเสียงที่แปลงแล้ว (ค่าเริ่มต้นอยู่ในโฟลเดอร์ `recordings/`)

## Other commands

```powershell
.\scripts\build_launcher.ps1   # build Parroty.exe (icon launcher)
.\scripts\make_shortcuts.ps1   # Desktop + Start menu shortcuts (-NoDesktop, -Remove)
.\run.ps1 -Diagnostics     # hardware / audio report
.\run.ps1 -CableTest       # test tone through VB-CABLE + 5 s robot-voice engine run
.\setup.ps1 -Dev; .\run.ps1 -Test   # unit tests
```

## Disk usage

| Install | Size |
|---|---|
| Lite (default) | ~330 MB (`.venv`, of which ~68 MB is PyAV/FFmpeg for WebM/M4A) |
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

Docs: **[User manual (Thai, HTML)](docs/MANUAL.html)**, also opened by the app's *Manual · คู่มือ* item (left pane) · [INSTALL_WINDOWS](docs/INSTALL_WINDOWS.md) · [TEAMS_SETUP](docs/TEAMS_SETUP.md) ·
[TROUBLESHOOTING](docs/TROUBLESHOOTING.md) · [MODEL_TRAINING](docs/MODEL_TRAINING.md)

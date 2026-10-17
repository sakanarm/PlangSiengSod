# Troubleshooting

Logs: `logs\rtvc.log` (rotating). Run `.\run.ps1 -Diagnostics` and `.\run.ps1 -CableTest`.

| Symptom | Cause / fix |
|---|---|
| Status *Not configured* | Read the reason next to the status. Usually no output selected → **Use VB-CABLE**. |
| "VB-CABLE was not found" | Install it (docs/INSTALL_WINDOWS.md), **reboot**, press *Refresh devices*. |
| "Feedback loop" error | Input is *CABLE Output*. Select your real mic as input. |
| Teams hears nothing | Teams microphone must be **CABLE Output**. App output must be **CABLE Input** (not "CABLE In 16ch"). Check app status is *Processing* and the output meter moves. |
| Input meter does not move | Windows mic privacy settings; wrong mic selected; mic muted (hardware key). |
| Robot voice is choppy in Teams | Teams *Noise suppression* → Off/Low. |
| Crackles / "underruns" counter rises | PC overloaded: close heavy apps; increase *output_prebuffer_blocks* in `config.json` (1 → 2). |
| "dropped" counter rises | Processing is slower than real time (shouldn't happen with the robot voice; report the log). |
| "Could not open audio device" | Another app uses the device in *exclusive mode*: Sound Control Panel → device → Advanced → untick *Allow applications to take exclusive control*. Or set 48000 Hz. |
| Error: device stopped/disconnected | USB/Bluetooth device removed. Reconnect → *Refresh devices* → Start. |
| Echo of my own voice | Monitor goes to speakers → use headphones or set Monitor to *(off)*. |
| Bluetooth headset sounds bad | Bluetooth switches to low-quality "Hands-Free" mode when its mic is used. Use the laptop mic or a wired/USB mic. |
| App will not start | Run `.\setup.ps1` again; check `logs\rtvc.log`. |
| Reset settings | Delete `config.json` (only contains your settings). |

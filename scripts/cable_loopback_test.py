"""Verify the VB-CABLE path on this PC with real devices (no files are written).

1. Tone test : play 440 Hz into 'CABLE Input' while recording 'CABLE Output'; checks the
               tone arrives (this is exactly what Teams will hear).
2. Engine run: microphone -> robot voice -> 'CABLE Input' for N seconds; prints the
               measured engine statistics (processing time, drops, underruns).

Usage:  .venv\\Scripts\\python scripts\\cable_loopback_test.py [--seconds 5] [--skip-engine]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtvc.audio import devices as dv  # noqa: E402
from rtvc.audio.engine import AudioEngine, play_and_record  # noqa: E402
from rtvc.audio.processors import RobotVoiceProcessor  # noqa: E402

API = "Windows WASAPI"
SR = 48000


def tone_test(cable_in: dv.Device, cable_out: dv.Device) -> bool:
    t = np.arange(SR) / SR
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    x = play_and_record(cable_in.index, cable_out.index, API, tone, SR)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    peak = float(freqs[np.argmax(spec)])
    level = 20 * np.log10(max(float(np.max(np.abs(x))), 1e-9))
    ok = abs(peak - 440) < 5 and level > -30
    print(f"[tone] recorded from '{cable_out.name}': peak {peak:.1f} Hz, level {level:.1f} dBFS -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def engine_test(mic: dv.Device, cable_in: dv.Device, seconds: float) -> bool:
    proc = RobotVoiceProcessor(SR, robot_pitch_hz=110)
    eng = AudioEngine(proc, hostapi=API, input_device=mic.index, output_device=cable_in.index,
                      samplerate=SR, block_time=0.05, prebuffer_blocks=1)
    eng.start()
    time.sleep(seconds)
    s = eng.stats
    print(f"[engine] {mic.name} -> robot -> {cable_in.name}, {seconds:.0f} s")
    print(f"  blocks in/out      : {s.blocks_in}/{s.blocks_out}")
    print(f"  processing ms      : avg {s.proc_ms_avg:.2f}, max {s.proc_ms_max:.2f}  (block = 50 ms, RTF {s.rtf:.3f})")
    print(f"  dropped / underruns: {s.dropped_blocks} / {s.underruns}   stream warnings: {s.stream_warnings}")
    print(f"  errors             : {s.processing_errors} {s.last_error}")
    print(f"  device latency     : {s.extra.get('device_latency_ms')}  est. app latency {s.est_latency_ms:.0f} ms")
    print(f"  last input level   : {s.input_db:.1f} dBFS")
    eng.stop()
    ok = s.blocks_out > 0 and s.processing_errors == 0
    print(f"[engine] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--skip-engine", action="store_true")
    a = ap.parse_args()
    ds = dv.list_devices()
    cable_in = next((d for d in dv.outputs(ds, API) if d.is_cable_playback and "16ch" not in d.name), None)
    cable_out = next((d for d in dv.inputs(ds, API) if d.is_cable_record), None)
    if not (cable_in and cable_out):
        print("VB-CABLE not found. Install it (see docs/INSTALL_WINDOWS.md) and reboot.")
        return 2
    ok = tone_test(cable_in, cable_out)
    if not a.skip_engine:
        mic = dv.find_device(ds, dv.default_device_name("input"), API, "input")
        if mic is None or mic.is_virtual:
            mic = next((d for d in dv.inputs(ds, API) if not d.is_virtual), None)
        if mic is None:
            print("No physical microphone found.")
            return 2
        ok = engine_test(mic, cable_in, a.seconds) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

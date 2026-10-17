"""Desktop UI (tkinter). Start with:  run.ps1   or   .venv\\Scripts\\python -m rtvc"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from .. import __version__, paths
from ..audio import devices as dv
from ..audio.engine import AudioEngine, play_test_tone
from ..audio.processors import ChainProcessor, PassthroughProcessor, RobotVoiceProcessor
from ..audio.voices import VOICES, key_for_label, make_voice
from ..config import AppConfig

logger = logging.getLogger(__name__)

STATE_COLORS = {"Not configured": "#b36b00", "Ready": "#1a7f37", "Processing": "#0b5cad", "Error": "#c62828"}
MONITOR_OFF = "(off)"
METER_FLOOR_DB = -60.0

TROUBLESHOOTING = {
    "not connected": "Plug the device back in, press 'Refresh devices', then Start again.",
    "Could not open": "Close other apps using the device exclusively, or check Windows Sound settings "
                      "(Privacy > Microphone access must be ON for desktop apps).",
    "Feedback": "Select your real microphone as input; 'CABLE Output' is for Teams, not for this app.",
    "Invalid sample rate": "Set the device to 48000 Hz in Windows Sound > device > Advanced.",
}


def _hint(msg: str) -> str:
    for key, hint in TROUBLESHOOTING.items():
        if key.lower() in msg.lower():
            return f"\n  -> {hint}"
    return "\n  -> See 'Manual / คู่มือ' (top right) > Troubleshooting"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = AppConfig.load(paths.CONFIG_FILE)
        bad = self.cfg.validate()
        if bad:
            logger.warning("Saved config invalid (%s); using defaults", bad)
            self.cfg = AppConfig()
        self.engine: AudioEngine | None = None
        self.devices: list[dv.Device] = []
        self._fatal_msg: str | None = None
        self.ai_available = all(importlib.util.find_spec(m) for m in ("torch", "onnxruntime"))
        try:
            import psutil

            self.proc = psutil.Process(os.getpid())
            self.proc.cpu_percent(None)
        except Exception:
            self.proc = None

        root.title(f"Parroty {__version__}")
        root.minsize(720, 620)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.refresh_devices(initial=True)
        self._poll()

    # ================================================================ layout
    def _build(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Status:").pack(side="left")
        self.state_lbl = tk.Label(top, text="Not configured", font=("Segoe UI", 11, "bold"))
        self.state_lbl.pack(side="left", padx=6)
        self.state_detail = ttk.Label(top, text="")
        self.state_detail.pack(side="left", padx=6)
        ttk.Button(top, text="Manual / คู่มือ", command=self.open_manual).pack(side="right")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        conv = ttk.Frame(nb, padding=10)
        hw = ttk.Frame(nb, padding=10)
        nb.add(conv, text="Voice")
        nb.add(hw, text="Hardware")
        self._build_convert(conv)
        self._build_hardware(hw)

    def _build_convert(self, f):
        # ---- voice mode
        mode = ttk.LabelFrame(f, text="Voice", padding=8)
        mode.pack(fill="x")
        self.mode = tk.StringVar(value=self.cfg.mode if self.cfg.mode != "rvc" or self.ai_available else "robot")
        ttk.Radiobutton(mode, text="Robot voice v1  (DSP effect, no AI model)", value="robot",
                        variable=self.mode, command=self._mode_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode, text="Other voices / เสียงอื่น  (DSP effect, no AI model)", value="voice",
                        variable=self.mode, command=self._mode_changed).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(mode, text="Pass-through  (no change - routing test)", value="passthrough",
                        variable=self.mode, command=self._mode_changed).grid(row=2, column=0, sticky="w")
        rvc = ttk.Radiobutton(mode, text="AI voice (RVC model)" + ("" if self.ai_available else
                              "  - add-on not installed (setup.ps1 -WithAI)"), value="rvc",
                              variable=self.mode, command=self._mode_changed)
        rvc.grid(row=3, column=0, sticky="nw")
        rvc.state(["disabled"])  # RVC streaming UI arrives with the AI add-on milestone
        mode.columnconfigure(1, weight=1)

        self.robot_frame = ttk.Frame(mode)
        self.robot_frame.grid(row=0, column=1, rowspan=4, sticky="ne", padx=(30, 0))
        self.robot_pitch = self._slider(self.robot_frame, 0, "Robot pitch (Hz)", 40, 400, self.cfg.robot_pitch_hz, 0)
        self.ring_hz = self._slider(self.robot_frame, 1, "Metallic wobble (Hz, 0=off)", 0, 200, self.cfg.ring_mod_hz, 0)
        self.ring_mix = self._slider(self.robot_frame, 2, "Wobble amount", 0, 1, self.cfg.ring_mix, 2)
        for v in (self.robot_pitch, self.ring_hz, self.ring_mix):
            v.trace_add("write", lambda *_: self._live_update())

        self.voice_frame = ttk.Frame(mode)
        self.voice_frame.grid(row=0, column=1, rowspan=4, sticky="ne", padx=(30, 0))
        ttk.Label(self.voice_frame, text="Voice / เสียง").grid(row=0, column=0, sticky="w")
        voice = VOICES.get(self.cfg.voice) or next(iter(VOICES.values()))
        self.voice_var = tk.StringVar(value=voice.label)
        self.voice_cb = ttk.Combobox(self.voice_frame, textvariable=self.voice_var, state="readonly", width=30,
                                     values=[v.label for v in VOICES.values()], height=len(VOICES))
        self.voice_cb.grid(row=0, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        self.voice_cb.bind("<<ComboboxSelected>>", lambda e: self._voice_selected())
        self.voice_pitch = self._slider(self.voice_frame, 1, "Pitch (semitones)", -12, 12, self.cfg.voice_pitch_st, 1)
        self.voice_formant = self._slider(self.voice_frame, 2, "Formant / voice size", -8, 8,
                                          self.cfg.voice_formant_st, 1)
        self.voice_desc = ttk.Label(self.voice_frame, text="", foreground="#555", wraplength=330)
        self.voice_desc.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))
        for v in (self.voice_pitch, self.voice_formant):
            v.trace_add("write", lambda *_: self._live_update())

        # ---- devices
        dev = ttk.LabelFrame(f, text="Audio routing", padding=8)
        dev.pack(fill="x", pady=8)
        dev.columnconfigure(1, weight=1)
        self.in_var, self.out_var, self.mon_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.in_cb = self._combo(dev, 0, "Microphone (input)", self.in_var)
        self.out_cb = self._combo(dev, 1, "Output -> Teams", self.out_var)
        self.mon_cb = self._combo(dev, 2, "Monitor (headphones)", self.mon_var)
        btns = ttk.Frame(dev)
        btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(btns, text="Refresh devices", command=self.refresh_devices).pack(side="left")
        ttk.Button(btns, text="Use VB-CABLE", command=self.auto_cable).pack(side="left", padx=6)
        ttk.Button(btns, text="Test beep -> output", command=lambda: self.test_tone("out")).pack(side="left")
        ttk.Button(btns, text="Test beep -> monitor", command=lambda: self.test_tone("mon")).pack(side="left", padx=6)

        # ---- levels
        lv = ttk.LabelFrame(f, text="Levels & output", padding=8)
        lv.pack(fill="x")
        lv.columnconfigure(1, weight=1)
        ttk.Label(lv, text="Input").grid(row=0, column=0, sticky="w")
        self.in_meter = tk.Canvas(lv, height=14, bg="#222", highlightthickness=0)
        self.in_meter.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(lv, text="Output").grid(row=1, column=0, sticky="w")
        self.out_meter = tk.Canvas(lv, height=14, bg="#222", highlightthickness=0)
        self.out_meter.grid(row=1, column=1, sticky="ew", padx=6, pady=2)
        g = ttk.Frame(lv)
        g.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.gain = self._slider(g, 0, "Output gain (dB)", -24, 12, self.cfg.output_gain_db, 0)
        self.gate = self._slider(g, 1, "Noise gate (dB, -60=off)", -60, 0, self.cfg.input_gate_db, 0)

        # ---- controls + stats
        ctl = ttk.Frame(f)
        ctl.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(ctl, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.rec_btn = ttk.Button(ctl, text="● Record", command=self.toggle_record, state="disabled")
        self.rec_btn.pack(side="left")
        self.stats_lbl = ttk.Label(ctl, text="", font=("Consolas", 9), justify="left")
        self.stats_lbl.pack(side="left", padx=12)

        ttk.Label(f, text="Be transparent: tell meeting participants you are using a voice effect.",
                  foreground="#555").pack(anchor="w")
        msg = ttk.LabelFrame(f, text="Messages", padding=4)
        msg.pack(fill="both", expand=True, pady=(6, 0))
        self.msg = tk.Text(msg, height=6, wrap="word", font=("Segoe UI", 9))
        self.msg.pack(fill="both", expand=True)
        self._mode_changed()

    def _build_hardware(self, f):
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        self.diag_btn = ttk.Button(bar, text="Run diagnostics", command=self.run_diagnostics)
        self.diag_btn.pack(side="left")
        self.diag_gpu_btn = ttk.Button(bar, text="Test GPU (DirectML) inference", command=lambda: self.run_diagnostics(True))
        self.diag_gpu_btn.pack(side="left", padx=6)
        if not self.ai_available:
            self.diag_gpu_btn.state(["disabled"])
        self.diag = tk.Text(f, wrap="none", font=("Consolas", 9))
        self.diag.pack(fill="both", expand=True, pady=(8, 0))
        self.diag.insert("end", "Press 'Run diagnostics'.")

    def _slider(self, parent, row, label, lo, hi, value, digits):
        var = tk.DoubleVar(value=value)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Scale(parent, from_=lo, to=hi, variable=var, length=180).grid(row=row, column=1, padx=6)
        val = ttk.Label(parent, width=6)
        val.grid(row=row, column=2, sticky="w")
        fmt = f"{{:.{digits}f}}"
        var.trace_add("write", lambda *_: val.configure(text=fmt.format(var.get())))
        val.configure(text=fmt.format(value))
        return var

    def _combo(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(parent, textvariable=var, state="readonly")
        cb.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_state())
        return cb

    # ================================================================ helpers
    def log(self, text: str, level=logging.INFO):
        logger.log(level, text)
        self.msg.insert("end", text + "\n")
        self.msg.see("end")

    def open_manual(self):
        """Open the local HTML manual in the default browser (offline file, nothing is fetched)."""
        import webbrowser

        if not paths.MANUAL_FILE.exists():
            self.log(f"Manual not found: {paths.MANUAL_FILE}", logging.WARNING)
            return
        if not webbrowser.open(paths.MANUAL_FILE.as_uri()):
            self.log(f"Could not open a browser. Open this file manually: {paths.MANUAL_FILE}", logging.WARNING)

    def set_state(self, state: str, detail: str = ""):
        self.state_lbl.configure(text=state, fg=STATE_COLORS[state])
        self.state_detail.configure(text=detail)

    def _mode_changed(self):
        m = self.mode.get()
        if m == "voice":
            self.robot_frame.grid_remove()
            self.voice_frame.grid()
        else:
            self.voice_frame.grid_remove()
            self.robot_frame.grid()
        on = m == "robot"
        for w in self.robot_frame.winfo_children():
            if isinstance(w, ttk.Scale):
                w.state(["!disabled"] if on else ["disabled"])
        self._voice_widgets()
        self._swap_processor()

    def _voice_widgets(self):
        v = VOICES[key_for_label(self.voice_var.get())]
        self.voice_desc.configure(text=v.description + ("" if v.adjustable else "  (no pitch/formant controls)"))
        for w in self.voice_frame.winfo_children():
            if isinstance(w, ttk.Scale):
                w.state(["!disabled"] if v.adjustable else ["disabled"])

    def _voice_selected(self):
        """A newly selected voice starts from its own default pitch/formant."""
        v = VOICES[key_for_label(self.voice_var.get())]
        self.voice_pitch.set(v.pitch_st)
        self.voice_formant.set(v.formant_st)
        self._voice_widgets()
        self._swap_processor()

    def _swap_processor(self):
        """While running, switching mode or voice takes effect immediately (new DSP chain)."""
        if not (self.engine and self.engine.stats.running) or self.mode.get() not in ("robot", "voice", "passthrough"):
            return
        try:
            proc = self._make_processor(self._collect_cfg())
        except Exception as e:
            self.log(f"Could not switch voice: {e}", logging.ERROR)
            return
        self.engine.processor = proc        # the worker thread uses it from the next block on
        self.set_state("Processing", proc.name)
        self.log(f"Voice switched to: {proc.name}")

    def _live_update(self):
        """Sliders apply immediately while running (only cheap parameters)."""
        if not (self.engine and self.engine.stats.running):
            return
        p = self.engine.processor
        if isinstance(p, RobotVoiceProcessor):
            p.ring_mod_hz = float(self.ring_hz.get())
            p.ring_mix = float(self.ring_mix.get())
        elif isinstance(p, ChainProcessor) and p.shift is not None:
            p.shift.pitch_st = float(self.voice_pitch.get())
            p.shift.formant_st = float(self.voice_formant.get())

    def _collect_cfg(self) -> AppConfig:
        c = self.cfg
        c.mode = self.mode.get()
        c.input_device, c.output_device = self.in_var.get(), self.out_var.get()
        c.monitor_device = "" if self.mon_var.get() in ("", MONITOR_OFF) else self.mon_var.get()
        c.robot_pitch_hz = round(float(self.robot_pitch.get()), 1)
        c.ring_mod_hz = round(float(self.ring_hz.get()), 1)
        c.ring_mix = round(float(self.ring_mix.get()), 2)
        c.voice = key_for_label(self.voice_var.get())
        c.voice_pitch_st = round(float(self.voice_pitch.get()), 1)
        c.voice_formant_st = round(float(self.voice_formant.get()), 1)
        c.output_gain_db = round(float(self.gain.get()), 1)
        c.input_gate_db = round(float(self.gate.get()), 1)
        return c

    def _update_state(self):
        if self.engine and self.engine.stats.running:
            return
        c = self._collect_cfg()
        errors, warnings = dv.validate_routing(c.input_device, c.output_device, c.monitor_device,
                                               self.devices, c.host_api)
        errors += c.validate()
        if errors:
            self.set_state("Not configured", errors[0])
        else:
            self.set_state("Ready", warnings[0] if warnings else "")

    # ================================================================ actions
    def refresh_devices(self, initial=False):
        if self.engine and self.engine.stats.running:
            self.log("Stop processing before refreshing devices.")
            return
        try:
            self.devices = dv.list_devices(rescan=not initial)
        except Exception as e:
            self.set_state("Error", "Audio system unavailable")
            self.log(f"Could not list audio devices: {e}", logging.ERROR)
            return
        api = self.cfg.host_api
        ins = [d.name for d in dv.inputs(self.devices, api)]
        outs = [d.name for d in dv.outputs(self.devices, api)]
        self.in_cb["values"] = ins
        self.out_cb["values"] = outs
        self.mon_cb["values"] = [MONITOR_OFF] + [n for n in outs if not any(h in n.lower() for h in dv.VIRTUAL_HINTS)]
        if initial:
            self.in_var.set(self.cfg.input_device or next(
                (d.name for d in dv.inputs(self.devices, api) if not d.is_virtual), ""))
            out = self.cfg.output_device
            if not out:
                cable = dv.suggest_cable_output(self.devices, api)
                out = cable.name if cable else ""
            self.out_var.set(out)
            self.mon_var.set(self.cfg.monitor_device or MONITOR_OFF)
            if not dv.cable_installed(self.devices):
                self.log("VB-CABLE was not found. Install it yourself from https://vb-audio.com/Cable/ "
                         "(administrator rights + reboot required) - see docs/INSTALL_WINDOWS.md.", logging.WARNING)
        else:
            self.log(f"Devices refreshed: {len(ins)} inputs, {len(outs)} outputs.")
        self._update_state()

    def auto_cable(self):
        cable = dv.suggest_cable_output(self.devices, self.cfg.host_api)
        if cable:
            self.out_var.set(cable.name)
            self.log(f"Output set to '{cable.name}'. In Teams select microphone 'CABLE Output'.")
        else:
            self.log("VB-CABLE not found - see docs/INSTALL_WINDOWS.md.", logging.WARNING)
        self._update_state()

    def test_tone(self, which):
        name = self.out_var.get() if which == "out" else self.mon_var.get()
        if not name or name == MONITOR_OFF:
            self.log("Choose a device first.")
            return
        if self.engine and self.engine.stats.running:
            self.log("Stop processing before the beep test.")
            return
        d = dv.find_device(self.devices, name, self.cfg.host_api, "output")
        if d is None:
            self.log(f"Device '{name}' not found.", logging.WARNING)
            return

        def run():
            try:
                play_test_tone(d.index, self.cfg.host_api, self.cfg.samplerate)
                self.root.after(0, self.log, f"Beep played on '{name}'."
                                + (" Teams 'Test call' / Windows input meter of 'CABLE Output' should move."
                                   if d.is_cable_playback else ""))
            except Exception as e:
                self.root.after(0, self.log, f"Beep failed on '{name}': {e}{_hint(str(e))}", logging.ERROR)

        threading.Thread(target=run, daemon=True).start()

    def _make_processor(self, c: AppConfig):
        if c.mode == "robot":
            return RobotVoiceProcessor(c.samplerate, c.robot_pitch_hz, c.ring_mod_hz, c.ring_mix)
        if c.mode == "voice":
            return make_voice(c.voice, c.samplerate, c.voice_pitch_st, c.voice_formant_st)
        if c.mode == "passthrough":
            return PassthroughProcessor()
        raise RuntimeError("AI voice mode is not available in this build.")

    def start(self):
        c = self._collect_cfg()
        errors, warnings = dv.validate_routing(c.input_device, c.output_device, c.monitor_device,
                                               self.devices, c.host_api)
        errors += c.validate()
        if errors:
            self.set_state("Not configured", errors[0])
            for e in errors:
                self.log("Cannot start: " + e + _hint(e), logging.WARNING)
            return
        for w in warnings:
            self.log("Warning: " + w)
        api = c.host_api
        ind = dv.find_device(self.devices, c.input_device, api, "input")
        outd = dv.find_device(self.devices, c.output_device, api, "output")
        mon = dv.find_device(self.devices, c.monitor_device, api, "output") if c.monitor_device else None
        try:
            proc = self._make_processor(c)
            self.engine = AudioEngine(proc, hostapi=api, input_device=ind.index, output_device=outd.index,
                                      monitor_device=mon.index if mon else None, samplerate=c.samplerate,
                                      block_time=c.dsp_block_time, prebuffer_blocks=c.output_prebuffer_blocks,
                                      input_gate_db=c.input_gate_db, output_gain_db=c.output_gain_db,
                                      on_fatal=self._on_fatal)
            self.engine.start()
        except Exception as e:
            self.engine = None
            self.set_state("Error", str(e)[:80])
            self.log(f"Start failed: {e}{_hint(str(e))}", logging.ERROR)
            return
        c.save(paths.CONFIG_FILE)
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self.rec_btn.state(["!disabled"])
        self.set_state("Processing", proc.name)
        self.log(f"Started: {ind.name} -> {proc.name} -> {outd.name}" + (f" (+ monitor {mon.name})" if mon else ""))

    def stop(self):
        if self.engine:
            self.stop_record()
            s = self.engine.stats
            self.engine.stop()
            self.log(f"Stopped. blocks={s.blocks_out} dropped={s.dropped_blocks} underruns={s.underruns} "
                     f"errors={s.processing_errors} avg processing={s.proc_ms_avg:.1f} ms")
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        self.rec_btn.state(["disabled"])
        self._update_state()

    # ---- recording of the processed voice (local WAV file, only on explicit request)
    def toggle_record(self):
        if self.engine and self.engine.recorder is not None:
            self.stop_record()
        else:
            self.start_record()

    def start_record(self):
        if not (self.engine and self.engine.stats.running):
            return
        paths.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save recording as", defaultextension=".wav",
            initialdir=str(paths.RECORDINGS_DIR),
            initialfile=f"rtvc_{datetime.now():%Y%m%d_%H%M%S}.wav",
            filetypes=[("WAV audio", "*.wav")])
        if not path or not (self.engine and self.engine.stats.running):
            return
        from ..audio.recorder import WavRecorder

        try:
            self.engine.recorder = WavRecorder(path, self.engine.sr)
        except Exception as e:
            self.log(f"Cannot record to {path}: {e}", logging.ERROR)
            return
        self.rec_btn.configure(text="■ Stop recording")
        self.log(f"Recording processed voice to {path}")

    def stop_record(self):
        rec = self.engine.recorder if self.engine else None
        if rec is None:
            return
        self.engine.recorder = None
        secs = rec.close()
        self.rec_btn.configure(text="● Record")
        extra = f", {rec.dropped} blocks skipped (disk too slow)" if rec.dropped else ""
        if rec.error:
            self.log(f"Recording failed: {rec.error}", logging.ERROR)
        else:
            self.log(f"Recording saved: {rec.path} ({secs:.1f} s{extra})")

    def _on_fatal(self, msg):  # called from engine threads
        self._fatal_msg = msg

    def run_diagnostics(self, verify_gpu=False):
        self.diag.delete("1.0", "end")
        self.diag.insert("end", "Running" + (" GPU inference test" if verify_gpu else "") + "...\n")
        for b in (self.diag_btn, self.diag_gpu_btn):
            b.state(["disabled"])

        def run():
            from ..hardware import diagnostics

            try:
                rep = diagnostics.collect(include_audio=not (self.engine and self.engine.stats.running))
                if verify_gpu:
                    rep.directml_verified = diagnostics.verify_directml(self.cfg.dml_device_id)[1]
                text = rep.text()
            except Exception as e:
                text = f"Diagnostics failed: {e}"
            logger.info("Diagnostics:\n%s", text)
            self.root.after(0, self._show_diag, text)

        threading.Thread(target=run, daemon=True).start()

    def _show_diag(self, text):
        self.diag.delete("1.0", "end")
        self.diag.insert("end", text)
        self.diag_btn.state(["!disabled"])
        if self.ai_available:
            self.diag_gpu_btn.state(["!disabled"])

    # ================================================================ periodic UI update
    def _meter(self, canvas: tk.Canvas, db: float):
        w = max(canvas.winfo_width(), 1)
        frac = min(max((db - METER_FLOOR_DB) / -METER_FLOOR_DB, 0.0), 1.0)
        canvas.delete("all")
        color = "#2e7d32" if db < -12 else "#f9a825" if db < -3 else "#c62828"
        canvas.create_rectangle(0, 0, int(w * frac), 14, fill=color, width=0)
        canvas.create_text(w - 4, 7, text=f"{db:.0f} dBFS" if db > -120 else "-inf", anchor="e",
                           fill="#ddd", font=("Segoe UI", 7))

    def _poll(self):
        if self._fatal_msg:
            msg, self._fatal_msg = self._fatal_msg, None
            self.stop()
            self.set_state("Error", msg[:90])
            self.log("ERROR: " + msg + _hint(msg), logging.ERROR)
        sys_txt = ""
        if self.proc:
            try:
                import psutil

                sys_txt = (f"CPU {psutil.cpu_percent(None):4.0f}% (app {self.proc.cpu_percent(None) / psutil.cpu_count():.0f}%)  "
                           f"RAM app {self.proc.memory_info().rss / 2**20:.0f} MB")
            except Exception:
                pass
        if self.engine and self.engine.stats.running:
            s = self.engine.stats
            self._meter(self.in_meter, s.input_db)
            self._meter(self.out_meter, s.output_db)
            rec = self.engine.recorder
            if rec is not None:
                self.rec_btn.configure(text=f"■ Stop recording ({rec.seconds:.0f} s)")
            self.stats_lbl.configure(text=(
                f"processing {s.proc_ms_avg:5.1f} ms (max {s.proc_ms_max:.1f})  RTF {s.rtf:.2f}  "
                f"latency ~{s.est_latency_ms:.0f} ms + device {s.extra.get('device_latency_ms', {})}\n"
                f"dropped {s.dropped_blocks}  underruns {s.underruns}  errors {s.processing_errors}  {sys_txt}"))
        else:
            self._meter(self.in_meter, -120)
            self._meter(self.out_meter, -120)
            self.stats_lbl.configure(text=sys_txt)
        self.root.after(200, self._poll)

    def on_close(self):
        try:
            if self.engine:
                self.engine.stop()
            self._collect_cfg().save(paths.CONFIG_FILE)
        except Exception:
            logger.exception("Error during shutdown")
        self.root.destroy()


def main():
    from logging.handlers import RotatingFileHandler

    paths.ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[RotatingFileHandler(paths.LOGS_DIR / "rtvc.log", maxBytes=2_000_000,
                                                      backupCount=3, encoding="utf-8"), logging.StreamHandler()])
    logger.info("Starting Parroty %s", __version__)
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    try:
        App(root)
    except Exception as e:
        logger.exception("UI failed to start")
        messagebox.showerror("Parroty", f"Failed to start: {e}")
        raise
    root.mainloop()

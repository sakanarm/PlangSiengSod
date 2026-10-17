"""'Voice from sample' page: measure a reference voice and save matched DSP settings as a new voice.

Everything shown here is measured (rtvc.audio.voice_match); "After matching" is the user's own
sample processed by the new voice and measured again. Audio stays in memory: recordings and
files are never copied, uploaded or stored with the voice.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from ..audio import custom_voices as cvs
from ..audio import devices as dv
from ..audio import voice_match as vm
from . import fluent as F

logger = logging.getLogger(__name__)

MAX_REC_S = 20.0
SLOTS = {"target": "Reference · ต้นแบบ", "own": "Your voice · เสียงคุณ"}


def _hz(v):
    return f"{v:.0f} Hz" if v else "—"


def _khz(h):
    return f"{h / 1000:g}k" if h >= 1000 else f"{h:g}"


class SamplePage:
    def __init__(self, app, page):
        self.app, self.page, self.fonts = app, page, app.fonts
        self.samples: dict[str, dict | None] = {"target": None, "own": None}
        self.result: vm.MatchResult | None = None
        self.recorder = None
        self._rec_slot = None
        self.player = None
        self._busy = False
        self._calls = queue.SimpleQueue()                   # worker thread -> UI thread
        self._build()
        self._drain()

    def _post(self, fn, *args):
        """Run ``fn(*args)`` on the UI thread (safe to call from a worker thread)."""
        self._calls.put((fn, args))

    def _drain(self):
        while True:
            try:
                fn, args = self._calls.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:
                logger.exception("UI update failed")
        self.app.root.after(100, self._drain)

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    # ================================================================ layout
    def _build(self):
        fn, page = self.fonts, self.page
        self.app._page_title(page, "Voice from sample · เสียงจากตัวอย่าง")
        F.label(page, "Measures the pitch, voice size (formants) and tone of a short recording and saves matching "
                      "settings as a new voice under 'My voices'. It is a DSP effect: the result sounds closer, "
                      "not like the same person.\n"
                      "วัดระดับเสียง ขนาดช่องเสียง (formant) และโทนเสียงจากตัวอย่าง แล้วบันทึกเป็นเสียงใหม่ "
                      "ได้เสียงที่ใกล้เคียงขึ้น ไม่ใช่เสียงของคนนั้น",
                fn.body, F.TEXT2, anchor="w", justify="left", wraplength=720).pack(fill="x", padx=(4, 28))

        rows = ctk.CTkFrame(page, fg_color="transparent")
        rows.pack(fill="x", padx=(0, 24), pady=(14, 0))
        r = F.SettingRow(rows, fn, F.I_INFO, "Permission · การยินยอม",
                         "The reference is my own voice, or the speaker allowed me to use it. I will not use the "
                         "result to pretend to be someone else.\nเป็นเสียงของฉันเอง หรือได้รับอนุญาตจากเจ้าของเสียงแล้ว "
                         "และจะไม่ใช้แอบอ้างเป็นผู้อื่น")
        r.pack(fill="x", pady=2)
        self.consent = tk.BooleanVar(value=False)
        F.checkbox(r.right, "I confirm · ยืนยัน", self.consent, self._update_buttons, fn).pack(side="right")

        self.slot_rows, self.rec_btns, self.file_btns = {}, {}, {}
        hints = {"target": "WAV, WebM, MP3, M4A, OGG, FLAC: 10–20 s of one person talking, little background noise.",
                 "own": "Speak normally for 10–20 s with the microphone you use in calls (set on Audio devices)."}
        for i, (slot, title) in enumerate(SLOTS.items(), 1):
            r = F.SettingRow(rows, fn, F.I_FILE if slot == "target" else F.I_MIC, f"{i}. {title}", hints[slot])
            r.pack(fill="x", pady=2)
            self.rec_btns[slot] = F.button(r.right, "●  Record", lambda s=slot: self.toggle_record(s), fn, width=150)
            self.rec_btns[slot].pack(side="right")
            self.file_btns[slot] = F.button(r.right, "Choose file…", lambda s=slot: self.choose_file(s), fn, width=130)
            self.file_btns[slot].pack(side="right", padx=(0, 8))
            self.slot_rows[slot] = r
        self._hints = hints

        self.analyze_row = F.SettingRow(rows, fn, F.I_PULSE, "3. Analyze · วิเคราะห์",
                                        "Needs both samples and the permission box. Takes a few seconds.")
        self.analyze_row.pack(fill="x", pady=2)
        self.analyze_btn = F.button(self.analyze_row.right, "Analyze", self.analyze, fn, accent=True, width=130)
        self.analyze_btn.pack(side="right")

        # ---- results (shown after an analysis)
        self.results = F.card(page)
        self.table = ctk.CTkFrame(self.results, fg_color="transparent")
        self.table.pack(fill="x", padx=18, pady=(14, 6))
        self.settings_lbl = F.label(self.results, "", fn.body_strong, anchor="w", justify="left", wraplength=720)
        self.settings_lbl.pack(fill="x", padx=18, pady=(6, 0))
        self.notes_lbl = F.label(self.results, "", fn.caption, F.CAUTION, anchor="w", justify="left", wraplength=720)
        self.notes_lbl.pack(fill="x", padx=18)
        F.label(self.results, "'After matching' is your own sample processed by the new voice and measured again "
                              "(a real measurement, not a prediction). Similar numbers do not mean the voices "
                              "sound the same.", fn.caption, F.TEXT2, anchor="w", justify="left",
                wraplength=720).pack(fill="x", padx=18, pady=(4, 14))

        self.after_rows = ctk.CTkFrame(page, fg_color="transparent")
        r = F.SettingRow(self.after_rows, fn, F.I_HEADPHONES, "4. Listen · ฟังเปรียบเทียบ",
                         "Plays on the Monitor (headphones) device, or on the Windows default output when the "
                         "Monitor is off. Never sent to Teams.")
        r.pack(fill="x", pady=2)
        self.play_btns = []
        for text, which in (("■", None), ("▶ Matched", "after"), ("▶ Yours", "own"), ("▶ Reference", "target")):
            b = F.button(r.right, text, lambda w=which: self.play(w), fn, width=44 if which is None else 110)
            b.pack(side="right", padx=(6, 0))
            self.play_btns.append(b)
        r = F.SettingRow(self.after_rows, fn, F.CUSTOM_ICON[0], "5. Save as a new voice · บันทึกเป็นเสียงใหม่",
                         "Only the numbers are saved (no audio). You can still move the sliders afterwards.",
                         symbol=True)
        r.pack(fill="x", pady=2)
        self.save_btn = F.button(r.right, "Save", self.save, fn, accent=True, width=90)
        self.save_btn.pack(side="right")
        self.name_entry = F.entry(r.right, fn, width=220, placeholder="Name, e.g. Narrator")
        self.name_entry.pack(side="right", padx=(0, 8))
        self._spacer = F.label(page, "", fn.caption)        # bottom spacing; results are packed above it
        self._spacer.pack(pady=8)
        self._update_buttons()

    # ================================================================ state
    def _update_buttons(self):
        ready = all(self.samples.values()) and self.consent.get() and not self._busy and not self.recording
        F.set_enabled(self.analyze_btn, ready)
        for slot in SLOTS:
            F.set_enabled(self.file_btns[slot], not self._busy and not self.recording)
            F.set_enabled(self.rec_btns[slot], not self._busy and (not self.recording or self._rec_slot == slot))

    def _set_sample(self, slot, audio, duration, source):
        self.samples[slot] = {"audio": audio, "duration": duration, "source": source}
        secs = audio.size / vm.SR
        cut = f" (first {secs:.0f} s used)" if duration > secs + 0.5 else ""
        self.slot_rows[slot].set_text(desc=f"✓ {source} · {duration:.1f} s{cut}\n{self._hints[slot]}")
        self._clear_result()
        self._update_buttons()

    def _clear_result(self):
        self.result = None
        self.stop_playback()
        self.results.pack_forget()
        self.after_rows.pack_forget()

    def _status(self, text):
        self.analyze_row.set_text(desc=text)

    # ================================================================ load / record
    def choose_file(self, slot):
        path = filedialog.askopenfilename(
            parent=self.app.root, title=f"Choose a voice sample – {SLOTS[slot]}",
            filetypes=[("Audio", " ".join("*" + e for e in vm.SUPPORTED_EXT)), ("All files", "*.*")])
        if not path:
            return
        self._busy = True
        self._update_buttons()
        self._status(f"Reading {Path(path).name}...")

        def run():
            try:
                audio, dur = vm.load_audio(path)
                self._post(self._loaded, slot, audio, dur, Path(path).name, None)
            except Exception as e:
                self._post(self._loaded, slot, None, 0, Path(path).name, e)

        threading.Thread(target=run, daemon=True).start()

    def _loaded(self, slot, audio, dur, name, err):
        self._busy = False
        if err is not None:
            self._status(f"Could not read {name}: {err}")
            self.app.log(f"Could not read the sample {name}: {err}", logging.WARNING)
        else:
            self._status("Needs both samples and the permission box. Takes a few seconds.")
            self._set_sample(slot, audio, dur, name)
        self._update_buttons()

    def toggle_record(self, slot):
        if self.recording:
            if self._rec_slot == slot:
                self._finish_record()
            return
        app = self.app
        if app.engine and app.engine.stats.running:
            app.log("Stop processing first: the microphone is in use by the voice changer.", logging.WARNING)
            self._status("Stop processing (Voice page) before recording a sample.")
            return
        mic = dv.find_device(app.devices, app.in_var.get(), app.cfg.host_api, "input")
        if mic is None:
            app.log("Choose your microphone on 'Audio devices' first.", logging.WARNING)
            self._status("Choose your microphone on 'Audio devices' first.")
            return
        from ..audio.clips import ClipRecorder

        try:
            self.recorder = ClipRecorder(mic.index, app.cfg.host_api, app.cfg.samplerate, MAX_REC_S)
        except Exception as e:
            app.log(f"Could not record from '{mic.name}': {e}", logging.ERROR)
            self._status(f"Could not record: {e}")
            return
        self._rec_slot = slot
        self._rec_mic = mic.name
        self.stop_playback()
        self._update_buttons()
        self._status(f"Recording from '{mic.name}' – speak normally. Stops by itself after {MAX_REC_S:.0f} s.")
        self._poll_record()

    def _poll_record(self):
        rec = self.recorder
        if rec is None:
            return
        if rec.full:
            self._finish_record()
            return
        self.rec_btns[self._rec_slot].configure(text=f"■  Stop ({rec.seconds:.0f} s)")
        self.app.root.after(200, self._poll_record)

    def _finish_record(self):
        rec, slot = self.recorder, self._rec_slot
        self.recorder = self._rec_slot = None
        self.rec_btns[slot].configure(text="●  Record")
        try:
            audio = rec.stop()
        except Exception as e:
            self.app.log(f"Recording failed: {e}", logging.ERROR)
            self._update_buttons()
            return
        if audio.size < rec.sr * 1.0:
            self._status("The recording is shorter than 1 s – please record again.")
            self._update_buttons()
            return
        peak = float(np.max(np.abs(audio)))
        audio = vm.resample(audio.astype(np.float64), rec.sr).astype(np.float32)
        self._set_sample(slot, audio, audio.size / vm.SR, f"Recording ({self._rec_mic})")
        if peak < 0.01:
            self._status(f"Very quiet recording (peak {20 * np.log10(peak + 1e-12):.0f} dBFS) – check the microphone.")
        else:
            self._status("Needs both samples and the permission box. Takes a few seconds.")

    # ================================================================ analysis
    def analyze(self):
        if not (all(self.samples.values()) and self.consent.get()) or self._busy:
            return
        self._busy = True
        self._clear_result()
        self._update_buttons()
        t, o = self.samples["target"], self.samples["own"]

        def run():
            try:
                res = vm.match(t["audio"], o["audio"], t["duration"], o["duration"],
                               progress=lambda s: self._post(self._status, s))
                self._post(self._show_result, res, None)
            except Exception as e:
                logger.exception("Voice analysis failed")
                self._post(self._show_result, None, e)

        threading.Thread(target=run, daemon=True).start()

    def _show_result(self, res, err):
        self._busy = False
        self._update_buttons()
        if err is not None:
            self._status(f"Analysis failed: {err}")
            self.app.log(f"Voice analysis failed: {err}", logging.ERROR)
            return
        self.result = res
        self._status("Done. Compare the values below, listen, then save.")
        fn = self.fonts
        for w in self.table.winfo_children():
            w.destroy()
        cols = ("", "Reference · ต้นแบบ", "Your voice · เสียงคุณ", "After matching · หลังปรับ")
        for c, text in enumerate(cols):
            F.label(self.table, text, fn.body_strong, anchor="w").grid(row=0, column=c, sticky="w", padx=(0, 18), pady=(0, 6))
        self.table.columnconfigure(0, weight=1)

        def fm(i):
            return lambda p: _hz(p.formants[i]) if p.formants else "—"

        rows = (
            ("Pitch, median · ระดับเสียง", lambda p: _hz(p.f0_median)),
            ("Pitch range (10–90 %)", lambda p: f"{p.f0_p10:.0f}–{p.f0_p90:.0f} Hz" if p.f0_p10 else "—"),
            ("Formant F1 · ขนาดช่องเสียง", fm(0)), ("Formant F2", fm(1)), ("Formant F3", fm(2)),
            ("Tone slope · ความลาดของโทน", lambda p: f"{p.tilt_db_oct:+.1f} dB/oct" if p.tilt_db_oct is not None else "—"),
            ("Brightness (spectral centroid)", lambda p: _hz(p.centroid_hz)),
            ("Speech with pitch · ช่วงที่มีเสียงพูด", lambda p: f"{p.voiced_s:.1f} of {p.duration_s:.1f} s"),
            ("Speech level", lambda p: f"{p.active_db:.0f} dBFS"),
        )
        for r, (name, get) in enumerate(rows, 1):
            F.label(self.table, name, fn.body, F.TEXT2, anchor="w").grid(row=r, column=0, sticky="w", padx=(0, 18))
            for c, prof in enumerate((res.target, res.own, res.after), 1):
                F.label(self.table, get(prof), fn.body, anchor="w").grid(row=r, column=c, sticky="w", padx=(0, 18))
        s = res.settings
        eq = ", ".join(f"{_khz(h)} {g:+.0f}" for h, g in s.eq if abs(g) >= 1) or "flat"
        self.settings_lbl.configure(text=f"New voice: pitch {s.pitch_st:+.1f} st  ·  formant {s.formant_st:+.1f} st"
                                         f"\nTone EQ (dB): {eq}")
        notes = list(s.notes) + [f"{SLOTS['target'].split(' ·')[0]}: {w}" for w in res.target.warnings] \
            + [f"{SLOTS['own'].split(' ·')[0]}: {w}" for w in res.own.warnings]
        self.notes_lbl.configure(text="\n".join("⚠ " + n for n in notes))
        self.results.pack(fill="x", padx=(0, 24), pady=(12, 0), before=self._spacer)
        self.after_rows.pack(fill="x", padx=(0, 24), pady=(10, 0), before=self._spacer)
        self.app.log(f"Voice sample analysed: pitch {s.pitch_st:+.1f} st, formant {s.formant_st:+.1f} st, "
                     f"EQ {eq}" + (f"; {len(notes)} warning(s)" if notes else ""))

    # ================================================================ preview
    def _preview_device(self):
        """(index or None for the host API default, name) for the preview, never the virtual cable."""
        import sounddevice as sd

        from .app import MONITOR_OFF

        app = self.app
        mon = app.mon_var.get()
        if mon and mon != MONITOR_OFF:
            d = dv.find_device(app.devices, mon, app.cfg.host_api, "output")
            if d is not None:
                return d.index, d.name
        for api in sd.query_hostapis():
            if api["name"] == app.cfg.host_api and api.get("default_output_device", -1) >= 0:
                idx = api["default_output_device"]
                name = sd.query_devices(idx)["name"]
                if any(h in name.lower() for h in dv.VIRTUAL_HINTS):
                    raise RuntimeError(f"the Windows default output is '{name}' (virtual cable); "
                                       "choose your headphones as Monitor on 'Audio devices'")
                return idx, name
        raise RuntimeError("no output device found; choose a Monitor device on 'Audio devices'")

    def play(self, which):
        self.stop_playback()
        if which is None or self.recording:
            return
        if which == "after":
            audio = self.result.after_audio if self.result else None
        else:
            audio = (self.samples.get(which) or {}).get("audio")
        if audio is None:
            return
        from ..audio.clips import ClipPlayer

        try:
            idx, name = self._preview_device()
            self.player = ClipPlayer(idx, self.app.cfg.host_api, audio, vm.SR)
        except Exception as e:
            self.app.log(f"Preview failed: {e}", logging.WARNING)
            self._status(f"Preview failed: {e}")
            return
        self._status(f"Playing on '{name}'...")

    def stop_playback(self):
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass
            self.player = None

    # ================================================================ save
    def save(self):
        res = self.result
        if res is None:
            return
        name = cvs.clean_name(self.name_entry.get())
        if not name:
            self._status("Enter a name for the new voice first.")
            return
        s = res.settings
        measured = {"reference": res.target.summary(), "own": res.own.summary(), "after": res.after.summary()}
        try:
            cv = cvs.create(name, s.pitch_st, s.formant_st, s.eq, measured)
        except (OSError, ValueError) as e:
            self._status(f"Could not save: {e}")
            self.app.log(f"Could not save the voice: {e}", logging.ERROR)
            return
        self.name_entry.delete(0, "end")
        self._status(f"Saved '{cv.name}'. It is selected on the Voice page.")
        self.app.add_custom_voice(cv)

    def close(self):
        self.stop_playback()
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception:
                pass
            self.recorder = None

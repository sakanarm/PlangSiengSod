"""Desktop UI (CustomTkinter, Windows 11 style). Start with:  run.ps1   or   .venv\\Scripts\\python -m rtvc"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import tkinter as tk
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np

from .. import __version__, paths
from ..audio import custom_voices as cvs
from ..audio.clips import ClipPlayer, ClipRecorder, render
from ..audio import devices as dv
from ..audio.engine import AudioEngine, play_test_tone
from ..audio.processors import ChainProcessor, PassthroughProcessor, RobotVoiceProcessor
from ..audio.voices import DEFAULT_VOICE, VOICES, get_voice, make_voice, voice_exists
from ..config import THEMES, AppConfig
from .sample_page import SamplePage
from . import fluent as F

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).parent / "assets"

STATE_COLORS = {"Not configured": F.CAUTION, "Ready": F.SUCCESS, "Processing": F.ACCENT, "Error": F.CRITICAL}
MONITOR_OFF = "(off)"
TEST_SECONDS = 5
TEST_HINT = ("Speak for 5 s, then hear it with the selected voice on your headphones or speakers. "
             "Nothing is sent to Teams. · พูด 5 วินาที แล้วฟังกลับเป็นเสียงที่เลือก (ไม่ส่งเข้า Teams)")
TILE_COLUMNS = 5
PAGES = (("voice", F.I_MIC, "Voice · เสียง"), ("sample", F.I_AUDIO, "Voice from sample · จากตัวอย่าง"),
         ("devices", F.I_SPEAKERS, "Audio devices · อุปกรณ์"),
         ("messages", F.I_MESSAGE, "Messages · ข้อความ"), ("hardware", F.I_PULSE, "Hardware"))

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
    return "\n  -> See 'Manual · คู่มือ' (left pane) > Troubleshooting"


def _split_label(label: str) -> tuple[str, str]:
    """'ชิปมังก์ / การ์ตูน (Chipmunk)' -> ('ชิปมังก์', 'Chipmunk');
    'แหบ (เสียงตัวเอง) (Hoarse, own voice)' -> ('แหบ (เสียงตัวเอง)', 'Hoarse, own voice')"""
    th, _, en = label.rpartition(" (")
    if not th:                                   # no English part
        return label.split(" / ")[0], ""
    return th.split(" / ")[0], en.removesuffix(")")


class App:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.cfg = AppConfig.load(paths.CONFIG_FILE)
        bad = self.cfg.validate()
        if bad:
            logger.warning("Saved config invalid (%s); using defaults", bad)
            self.cfg = AppConfig()
        self.engine: AudioEngine | None = None
        self.devices: list[dv.Device] = []
        self._fatal_msg: str | None = None
        self._unread = 0
        self._stats_text = ""
        self._test_rec: ClipRecorder | None = None     # "Test my voice": recording in progress
        self._test_take = None                          # last raw take (mic audio, memory only)
        self._test_player: ClipPlayer | None = None
        self._test_busy = False                         # rendering in a worker thread
        self.ai_available = all(importlib.util.find_spec(m) for m in ("torch", "onnxruntime"))
        try:
            import psutil

            self.proc = psutil.Process(os.getpid())
            self.proc.cpu_percent(None)
        except Exception:
            self.proc = None

        ctk.set_appearance_mode(self.cfg.theme)
        self.fonts = F.Fonts()
        root.title("Parroty")
        try:
            root.iconbitmap(default=str(ASSETS / "parroty.ico"))   # title bar + taskbar
        except tk.TclError:
            logger.warning("App icon not found: %s", ASSETS / "parroty.ico")
        root.configure(fg_color=F.WINDOW)
        # fits a 1080p screen at 125 % scaling with the taskbar; centred in the upper part of the screen
        scale = ctk.ScalingTracker.get_window_scaling(root)
        w, h = 1100, 700
        x = max(0, int((root.winfo_screenwidth() / scale - w) / 2))
        y = max(0, int((root.winfo_screenheight() / scale - h - 48 - 32) / 2))
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(900, 560)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.refresh_devices(initial=True)
        self._poll()

    # ================================================================ layout
    def _build(self):
        fn = self.fonts
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ---- navigation pane
        nav = ctk.CTkFrame(self.root, fg_color="transparent", width=250)
        nav.grid(row=0, column=0, sticky="ns", padx=(8, 0), pady=8)
        nav.pack_propagate(False)
        brand = ctk.CTkFrame(nav, fg_color="transparent")
        brand.pack(fill="x", padx=8, pady=(8, 18))
        self._logo = self._load_logo()
        if self._logo is not None:
            with warnings.catch_warnings():      # CTk warns about non-CTkImage images (no Pillow here)
                warnings.simplefilter("ignore")
                ctk.CTkLabel(brand, text="", image=self._logo, width=32, height=32).pack(side="left")
        else:
            F.label(brand, F.I_MIC, fn.icon(16), F.ON_ACCENT, fg_color=F.ACCENT, corner_radius=6,
                    width=30, height=30).pack(side="left")
        F.label(brand, "Parroty", fn.body_strong).pack(side="left", padx=(10, 6))
        F.label(brand, "v" + ".".join(__version__.split(".")[:2]) + " by Sakan Punyanon", fn.caption,
                F.TEXT2).pack(side="left", pady=(2, 0))
        self.nav = {}
        for key, icon, text in PAGES:
            item = F.NavItem(nav, fn, icon, text, lambda k=key: self.show_page(k))
            item.pack(fill="x", pady=1)
            self.nav[key] = item
        self.theme_item = F.NavItem(nav, fn, F.I_THEME, "", self.cycle_theme)
        self.theme_item.pack(side="bottom", fill="x", pady=1)
        F.NavItem(nav, fn, F.I_HELP, "Manual · คู่มือ", self.open_manual).pack(side="bottom", fill="x", pady=1)
        self._theme_text()

        # ---- pages
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        self.pages = {}
        for key in ("voice", "sample"):
            self.pages[key] = ctk.CTkScrollableFrame(
                content, fg_color="transparent", scrollbar_button_color=F.TRACK, scrollbar_button_hover_color=F.TEXT2)
        for key in ("devices", "messages", "hardware"):
            self.pages[key] = ctk.CTkFrame(content, fg_color="transparent")
        for p in self.pages.values():
            p.grid(row=0, column=0, sticky="nsew")
        self._build_voice(self.pages["voice"])
        self.sample_page = SamplePage(self, self.pages["sample"])
        self._build_devices(self.pages["devices"])
        self._build_messages(self.pages["messages"])
        self._build_hardware(self.pages["hardware"])
        self.show_page("voice")

    def _page_title(self, page, text):
        F.label(page, text, self.fonts.title, anchor="w").pack(fill="x", padx=(4, 28), pady=(18, 14))

    def _section(self, page, title, caption=""):
        F.label(page, title, self.fonts.body_strong, anchor="w").pack(fill="x", padx=(4, 28), pady=(22, 0))
        if caption:
            F.label(page, caption, self.fonts.caption, F.TEXT2, anchor="w").pack(fill="x", padx=(4, 28))

    def _build_voice(self, page):
        fn = self.fonts
        self._page_title(page, "Voice · เสียง")

        # ---- status / start card
        hero = F.card(page)
        hero.pack(fill="x", padx=(0, 24))
        hero.columnconfigure(0, weight=1)
        top = ctk.CTkFrame(hero, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        top.columnconfigure(2, weight=1)
        self.state_dot = F.label(top, "●", (fn.body[0], 16), F.CAUTION, width=16)
        self.state_dot.grid(row=0, column=0, padx=(0, 8))
        self.state_lbl = F.label(top, "Not configured", fn.subtitle, anchor="w")
        self.state_lbl.grid(row=0, column=1, sticky="w")
        self.rec_btn = F.button(top, "●  Record", self.toggle_record, fn, width=150)
        self.rec_btn.grid(row=0, column=3, padx=(8, 0))
        F.set_enabled(self.rec_btn, False)
        self.run_btn = F.button(top, "▶  Start", self.toggle_run, fn, accent=True, width=130)
        self.run_btn.grid(row=0, column=4, padx=(8, 0))
        info = ctk.CTkFrame(hero, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew", padx=(44, 20))
        self.state_detail = F.label(info, "", fn.body, F.TEXT2, anchor="w", justify="left", wraplength=560)
        self.state_detail.pack(side="left")
        self.state_link = F.link(info, "", lambda: self.show_page(self._link_page), fn)
        self._link_page = "devices"
        meters = ctk.CTkFrame(hero, fg_color="transparent")
        meters.grid(row=2, column=0, sticky="ew", padx=20, pady=(12, 0))
        self.in_meter = F.Meter(meters, fn, "Input")
        self.in_meter.pack(fill="x", pady=2)
        self.out_meter = F.Meter(meters, fn, "Output")
        self.out_meter.pack(fill="x", pady=2)
        self.stats_lbl = F.label(hero, "", fn.caption, F.TEXT2, anchor="w", justify="left")
        self.stats_lbl.grid(row=3, column=0, sticky="w", padx=20, pady=(6, 14))

        # ---- test my voice: record a few seconds, hear them back with the selected voice (local only)
        self.test_row = F.SettingRow(page, fn, F.I_HEADPHONES, "Test my voice · ทดสอบเสียง", TEST_HINT)
        self.test_row.pack(fill="x", padx=(0, 24), pady=(8, 0))
        self.replay_btn = F.button(self.test_row.right, "▶  Play again", self.replay_voice_test, fn, width=130)
        self.replay_btn.pack(side="right", padx=(8, 0))
        F.set_enabled(self.replay_btn, False)
        self.test_btn = F.button(self.test_row.right, f"●  Speak {TEST_SECONDS} s", self.toggle_voice_test, fn,
                                 width=140)
        self.test_btn.pack(side="right")

        # ---- voice tiles
        self.mode = tk.StringVar(value=self.cfg.mode if self.cfg.mode != "rvc" or self.ai_available else "robot")
        self.voice_key = self.cfg.voice if voice_exists(self.cfg.voice) else DEFAULT_VOICE
        self._section(page, "Choose a voice · เลือกเสียง",
                      "Sound effects (DSP) – no AI model, runs on the CPU. You can switch while running.")
        people = [k for k, v in VOICES.items() if v.group == "person"] + ["passthrough"]
        effects = ["robot"] + [k for k, v in VOICES.items() if v.group == "effect"]
        self.tiles = {}
        for title, keys in (("People · เสียงคน", people), ("Effects · เอฟเฟกต์", effects)):
            F.label(page, title, fn.caption, F.TEXT2, anchor="w").pack(fill="x", padx=4, pady=(10, 4))
            grid = ctk.CTkFrame(page, fg_color="transparent")
            grid.pack(fill="x", padx=(0, 24))
            self._tile_grid(grid, keys)
        F.label(page, "My voices · เสียงของฉัน (from a sample)", fn.caption, F.TEXT2,
                anchor="w").pack(fill="x", padx=4, pady=(10, 4))
        self.custom_box = ctk.CTkFrame(page, fg_color="transparent")
        self.custom_box.pack(fill="x", padx=(0, 24))
        self.rebuild_custom_tiles()
        if not self.ai_available:
            F.label(page, "AI voice (RVC model): add-on not installed – not available in this version.",
                    fn.caption, F.TEXT_DISABLED, anchor="w").pack(fill="x", padx=4, pady=(8, 0))

        # ---- adjust
        self._section(page, "Adjust · ปรับแต่ง", "Changes apply immediately, also while running.")
        self.adjust = ctk.CTkFrame(page, fg_color="transparent")
        self.adjust.pack(fill="x", padx=(0, 24), pady=(6, 0))
        self.sel_row = F.SettingRow(self.adjust, fn, F.I_MIC, "", "")
        self.reset_btn = F.button(self.sel_row.right, "Reset to preset", self._reset_voice, fn, width=140)
        self.reset_btn.pack(side="right")
        self.delete_btn = F.button(self.sel_row.right, "Delete voice", self.delete_custom_voice, fn, width=120)

        def row(icon, title, desc, lo, hi, value, digits, steps, unit=""):
            r = F.SettingRow(self.adjust, fn, icon, title, desc)
            var, s = F.slider(r.right, fn, lo, hi, value, digits, steps, unit)
            var.trace_add("write", lambda *_: self._live_update())
            return r, var, s

        r1, self.robot_pitch, _ = row(F.I_ROBOT, "Robot pitch (Hz)", "Pitch of the robot voice. Applies at the next Start.",
                                      40, 400, self.cfg.robot_pitch_hz, 0, 360, " Hz")
        r2, self.ring_hz, s2 = row(F.I_RADIO, "Metallic wobble (Hz, 0=off)", "Speed of the metallic wobble.",
                                   0, 200, self.cfg.ring_mod_hz, 0, 200, " Hz")
        r3, self.ring_mix, s3 = row(F.I_VOLUME, "Wobble amount", "Strength of the wobble.", 0, 1, self.cfg.ring_mix, 2, 100)
        self.robot_rows = (r1, r2, r3)
        r4, self.voice_pitch, s4 = row(F.I_MUSIC, "Pitch (semitones)", "+12 = one octave higher, −12 = one octave lower.",
                                       -12, 12, self.cfg.voice_pitch_st, 1, 48, " st")
        r5, self.voice_formant, s5 = row(F.I_EMOJI, "Formant / voice size",
                                         "Size of the speaker you hear: − = bigger / deeper, + = smaller. Pitch stays.",
                                         -8, 8, self.cfg.voice_formant_st, 1, 32, " st")
        self.voice_rows, self.voice_sliders = (r4, r5), (s4, s5)
        self.eq_row, self.voice_eq, _ = row(F.I_VOLUME, "Tone match (EQ amount)",
                                            "How much of the measured tone (EQ) is applied: 0 = off, 1 = full.",
                                            0, 1, self.cfg.voice_eq_amount, 2, 20)

        self._section(page, "Output · เสียงออก")
        out = ctk.CTkFrame(page, fg_color="transparent")
        out.pack(fill="x", padx=(0, 24), pady=(6, 0))
        r = F.SettingRow(out, fn, F.I_VOLUME, "Output gain (dB)", "Lower it if the Output meter turns red.")
        r.pack(fill="x", pady=2)
        self.gain, _ = F.slider(r.right, fn, -24, 12, self.cfg.output_gain_db, 1, 72, " dB")
        r = F.SettingRow(out, fn, F.I_MIC, "Noise gate (dB, −60 = off)",
                         "Mutes sounds quieter than this, e.g. a fan. Try −45; too high cuts your words.")
        r.pack(fill="x", pady=2)
        self.gate, _ = F.slider(r.right, fn, -60, 0, self.cfg.input_gate_db, 0, 60, " dB")

        bar = ctk.CTkFrame(page, fg_color=F.INFO_BG, corner_radius=7)
        bar.pack(fill="x", padx=(0, 24), pady=(22, 24))
        F.label(bar, F.I_INFO, fn.icon(16), F.ACCENT).pack(side="left", padx=(16, 10), pady=12)
        F.label(bar, "Be transparent: tell meeting participants you are using a voice effect. "
                     "Do not use it to pretend to be someone else.", fn.body, anchor="w",
                justify="left", wraplength=640).pack(side="left", pady=12)
        self._mode_changed()

    def _tile_grid(self, grid, keys):
        """Voice tiles, TILE_COLUMNS per row (rows wrap so the window can stay narrow)."""
        for c in range(TILE_COLUMNS):
            grid.columnconfigure(c, weight=1, uniform="tile")
        for i, key in enumerate(keys):
            row, col = divmod(i, TILE_COLUMNS)
            th, en = self._tile_text(key)
            icon, sym = self._icon(key)
            t = F.Tile(grid, self.fonts, icon, sym, th, en, lambda k=key: self.select_voice(k))
            t.grid(row=row, column=col, sticky="nsew", pady=(0 if row == 0 else 6, 0),
                   padx=(0 if col == 0 else 3, 3 if col < TILE_COLUMNS - 1 else 0))
            self.tiles[key] = t

    def rebuild_custom_tiles(self):
        """(Re)create the 'My voices' tiles from the saved custom voices."""
        for w in self.custom_box.winfo_children():
            w.destroy()
        for k in [k for k in self.tiles if cvs.is_custom_key(k)]:
            del self.tiles[k]
        keys = [cv.key for cv in cvs.list_voices()]
        if keys:
            grid = ctk.CTkFrame(self.custom_box, fg_color="transparent")
            grid.pack(fill="x")
            self._tile_grid(grid, keys)
        else:
            empty = ctk.CTkFrame(self.custom_box, fg_color="transparent")
            empty.pack(fill="x")
            F.label(empty, "None yet. Make one from a voice sample:", self.fonts.caption, F.TEXT2).pack(side="left", padx=4)
            F.link(empty, "Voice from sample ›", lambda: self.show_page("sample"), self.fonts).pack(side="left")
        sel = self._selected_key()
        for k, t in self.tiles.items():
            t.set_selected(k == sel)

    def _icon(self, key):
        return F.CUSTOM_ICON if cvs.is_custom_key(key) else F.VOICE_ICONS.get(key, (F.I_MIC, False))

    def _tile_text(self, key):
        if key == "robot":
            return "หุ่นยนต์", "Robot"
        if key == "passthrough":
            return "เสียงจริง", "Pass-through"
        return _split_label(get_voice(key).label)

    def _build_devices(self, page):
        fn = self.fonts
        self._page_title(page, "Audio devices · อุปกรณ์เสียง")
        self.in_var, self.out_var, self.mon_var = tk.StringVar(), tk.StringVar(), tk.StringVar()

        def row(icon, title, desc, var):
            r = F.SettingRow(page, fn, icon, title, desc)
            r.pack(fill="x", padx=(0, 28), pady=2)
            box, om = F.option_menu(r.right, var, lambda _v: self._update_state(), fn)
            box.pack(side="right")
            return r, om

        _, self.in_cb = row(F.I_MIC, "Microphone (input)", "Your real microphone. Never choose CABLE Output here.",
                            self.in_var)
        r, self.out_cb = row(F.I_SPEAKERS, "Output → Teams",
                             "CABLE Input (VB-Audio Virtual Cable). In Teams choose the microphone CABLE Output.",
                             self.out_var)
        F.button(r.right, "Use VB-CABLE", self.auto_cable, fn).pack(side="right", padx=(0, 8))
        _, self.mon_cb = row(F.I_HEADPHONES, "Monitor (headphones)",
                             "Optional: hear your converted voice. Use headphones, not speakers.", self.mon_var)

        r = F.SettingRow(page, fn, F.I_VOLUME, "Test the route",
                         "Plays a short beep. Stop processing first. 'Output' goes into VB-CABLE for Teams, so you "
                         "will not hear it; 'monitor' plays on your headphones. To hear your voice, use "
                         "'Test my voice' on the Voice page.")
        r.pack(fill="x", padx=(0, 28), pady=(14, 2))
        F.button(r.right, "Test beep → monitor", lambda: self.test_tone("mon"), fn, width=170).pack(side="right")
        F.button(r.right, "Test beep → output", lambda: self.test_tone("out"), fn, width=170).pack(side="right", padx=(0, 8))
        r = F.SettingRow(page, fn, F.I_SETTINGS, "Refresh devices", "After plugging in a microphone or headphones.")
        r.pack(fill="x", padx=(0, 28), pady=2)
        F.button(r.right, "Refresh devices", self.refresh_devices, fn, width=150).pack(side="right")

    def _build_messages(self, page):
        fn = self.fonts
        self._page_title(page, "Messages · ข้อความ")
        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.pack(fill="x", padx=(0, 28), pady=(0, 8))
        F.button(bar, "Clear", lambda: self.msg.delete("1.0", "end"), fn, width=90).pack(side="left")
        F.button(bar, "Open log folder", self.open_logs, fn, width=150).pack(side="left", padx=8)
        self.msg = ctk.CTkTextbox(page, font=fn.body, wrap="word", fg_color=F.CARD, border_color=F.BORDER,
                                  border_width=1, corner_radius=7, text_color=F.TEXT,
                                  scrollbar_button_color=F.TRACK, scrollbar_button_hover_color=F.TEXT2)
        self.msg.pack(fill="both", expand=True, padx=(0, 28), pady=(0, 24))

    def _build_hardware(self, page):
        fn = self.fonts
        self._page_title(page, "Hardware")
        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.pack(fill="x", padx=(0, 28), pady=(0, 8))
        self.diag_btn = F.button(bar, "Run diagnostics", self.run_diagnostics, fn, accent=True, width=150)
        self.diag_btn.pack(side="left")
        self.diag_gpu_btn = F.button(bar, "Test GPU (DirectML) inference", lambda: self.run_diagnostics(True), fn,
                                     width=240)
        self.diag_gpu_btn.pack(side="left", padx=8)
        if not self.ai_available:
            F.set_enabled(self.diag_gpu_btn, False)
        self.diag = ctk.CTkTextbox(page, font=fn.mono, wrap="none", fg_color=F.CARD, border_color=F.BORDER,
                                   border_width=1, corner_radius=7, text_color=F.TEXT,
                                   scrollbar_button_color=F.TRACK, scrollbar_button_hover_color=F.TEXT2)
        self.diag.pack(fill="both", expand=True, padx=(0, 28), pady=(0, 24))
        self.diag.insert("end", "Press 'Run diagnostics'.")

    # ================================================================ helpers
    def _load_logo(self):
        """Parroty logo for the nav pane, pre-sized for the display scaling (tk PhotoImage reads PNG)."""
        want = 32 * ctk.ScalingTracker.get_widget_scaling(self.root)
        size = min((32, 40, 48, 64), key=lambda px: abs(px - want))
        try:
            return tk.PhotoImage(master=self.root, file=str(ASSETS / f"parroty_{size}.png"))
        except tk.TclError:
            return None

    def show_page(self, key):
        for k, p in self.pages.items():
            if k == key:
                p.grid()
            else:
                p.grid_remove()
        for k, item in self.nav.items():
            item.set_selected(k == key)
        self.page = key
        if key == "messages":
            self._unread = 0
            self.nav["messages"].text.configure(text="Messages · ข้อความ")

    def cycle_theme(self):
        self.cfg.theme = THEMES[(THEMES.index(self.cfg.theme) + 1) % len(THEMES)]
        ctk.set_appearance_mode(self.cfg.theme)
        self._theme_text()

    def _theme_text(self):
        self.theme_item.text.configure(text=f"Theme: {self.cfg.theme.capitalize()}")

    def log(self, text: str, level=logging.INFO):
        logger.log(level, text)
        self.msg.insert("end", datetime.now().strftime("%H:%M:%S  ") + text + "\n")
        self.msg.see("end")
        if level >= logging.WARNING and getattr(self, "page", "") != "messages":
            self._unread += 1
            self.nav["messages"].text.configure(text=f"Messages · ข้อความ  ({self._unread})")

    def open_manual(self):
        """Open the local HTML manual in the default browser (offline file, nothing is fetched)."""
        import webbrowser

        if not paths.MANUAL_FILE.exists():
            self.log(f"Manual not found: {paths.MANUAL_FILE}", logging.WARNING)
            return
        if not webbrowser.open(paths.MANUAL_FILE.as_uri()):
            self.log(f"Could not open a browser. Open this file manually: {paths.MANUAL_FILE}", logging.WARNING)

    def open_logs(self):
        paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(paths.LOGS_DIR)  # noqa: S606 - local folder in Explorer

    def set_state(self, state: str, detail: str = ""):
        self.state_dot.configure(text_color=STATE_COLORS[state])
        self.state_lbl.configure(text=state)
        self.state_detail.configure(text=detail)
        link = {"Not configured": ("devices", "Audio devices ›"), "Error": ("messages", "Messages ›")}.get(state)
        if link:
            self._link_page = link[0]
            self.state_link.configure(text=link[1])
            self.state_link.pack(side="left", padx=(8, 0))
        else:
            self.state_link.pack_forget()

    def _selected_key(self) -> str:
        m = self.mode.get()
        return self.voice_key if m == "voice" else m

    def select_voice(self, key: str):
        """Tile click. A voice starts from its own default pitch/formant (and tone match)."""
        if key in ("robot", "passthrough"):
            self.mode.set(key)
        else:
            try:
                v = get_voice(key)
            except ValueError as e:
                self.log(f"Cannot select voice: {e}", logging.WARNING)
                return
            self.mode.set("voice")
            self.voice_key = key
            self.voice_pitch.set(v.pitch_st)
            self.voice_formant.set(v.formant_st)
            self.voice_eq.set(v.eq_amount)
        self._mode_changed()

    def _reset_voice(self):
        v = get_voice(self.voice_key)
        self.voice_pitch.set(v.pitch_st)
        self.voice_formant.set(v.formant_st)
        self.voice_eq.set(v.eq_amount)

    def _mode_changed(self):
        key = self._selected_key()
        if key not in ("robot", "passthrough") and not voice_exists(key):
            self.voice_key = key = DEFAULT_VOICE              # e.g. a custom voice file was removed
        for k, t in self.tiles.items():
            t.set_selected(k == key)
        icon, sym = self._icon(key)
        th, en = self._tile_text(key)
        rows = [self.sel_row]
        custom = cvs.is_custom_key(key)
        if key == "robot":
            desc = "Classic robot effect with a fixed pitch and an optional metallic wobble."
            rows += self.robot_rows
        elif key == "passthrough":
            desc = "No change: your real voice is sent. Use it to test the route to Teams."
        else:
            v = get_voice(key)
            th = v.label.partition(" (")[0]
            desc = v.description + ("" if v.adjustable else "  No pitch / formant controls for this voice.")
            rows += self.voice_rows
            if v.eq:
                rows.append(self.eq_row)
            for s in self.voice_sliders:
                F.set_enabled(s, v.adjustable)
            F.set_enabled(self.reset_btn, v.adjustable)
        self.sel_row.set_text(f"{th} · {en}", desc, icon, self.fonts, sym)
        if key in ("robot", "passthrough"):
            self.reset_btn.pack_forget()
        else:
            self.reset_btn.pack(side="right")
        if custom:
            self.delete_btn.pack(side="right", padx=(0, 8))
        else:
            self.delete_btn.pack_forget()
        for r in (self.sel_row, *self.robot_rows, *self.voice_rows, self.eq_row):
            r.pack_forget()
        for r in rows:
            r.pack(fill="x", pady=2)
        self._swap_processor()

    def add_custom_voice(self, cv: cvs.CustomVoice):
        """Called by the sample page after saving: show the new tile and select it."""
        self.rebuild_custom_tiles()
        self.select_voice(cv.key)
        self.show_page("voice")
        self.log(f"New voice '{cv.name}' saved (pitch {cv.pitch_st:+.1f} st, formant {cv.formant_st:+.1f} st, "
                 f"EQ {len(cv.eq)} bands) in {paths.CUSTOM_VOICES_DIR}")

    def delete_custom_voice(self):
        key = self.voice_key
        if not cvs.is_custom_key(key):
            return
        name = get_voice(key).label.partition(" (")[0]
        if not messagebox.askyesno("Delete voice", f"Delete the voice '{name}'?\nThis cannot be undone.",
                                   parent=self.root):
            return
        try:
            cvs.delete(key[len(cvs.PREFIX):])
        except OSError as e:
            self.log(f"Could not delete '{name}': {e}", logging.ERROR)
            return
        self.log(f"Voice '{name}' deleted.")
        self.voice_key = DEFAULT_VOICE
        self.rebuild_custom_tiles()
        self.select_voice(DEFAULT_VOICE)

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
            if p.eq is not None:
                p.eq.amount = float(self.voice_eq.get())

    def _collect_cfg(self) -> AppConfig:
        c = self.cfg
        c.mode = self.mode.get()
        c.input_device, c.output_device = self.in_var.get(), self.out_var.get()
        c.monitor_device = "" if self.mon_var.get() in ("", MONITOR_OFF) else self.mon_var.get()
        c.robot_pitch_hz = round(float(self.robot_pitch.get()), 1)
        c.ring_mod_hz = round(float(self.ring_hz.get()), 1)
        c.ring_mix = round(float(self.ring_mix.get()), 2)
        c.voice = self.voice_key
        c.voice_pitch_st = round(float(self.voice_pitch.get()), 1)
        c.voice_formant_st = round(float(self.voice_formant.get()), 1)
        c.voice_eq_amount = round(float(self.voice_eq.get()), 2)
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
            self.set_state("Ready", warnings[0] if warnings else "Press Start, then choose 'CABLE Output' as the microphone in Teams.")

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
        self.in_cb.configure(values=ins)
        self.out_cb.configure(values=outs)
        self.mon_cb.configure(values=[MONITOR_OFF] + [n for n in outs if not any(h in n.lower() for h in dv.VIRTUAL_HINTS)])
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
            self.log("Choose the 'Output → Teams' device first." if which == "out" else
                     "Monitor is off: choose your headphones under 'Monitor (headphones)' first.", logging.WARNING)
            return
        if self.engine and self.engine.stats.running:
            self.log("Stop processing before the beep test.", logging.WARNING)
            return
        d = dv.find_device(self.devices, name, self.cfg.host_api, "output")
        if d is None:
            self.log(f"Device '{name}' not found.", logging.WARNING)
            return

        def run():
            try:
                play_test_tone(d.index, self.cfg.host_api, self.cfg.samplerate)
                self.root.after(0, self.log, f"Beep played on '{name}'."
                                + (" It goes into VB-CABLE, so you do not hear it: Teams 'Test call' / the "
                                   "Windows input meter of 'CABLE Output' should move."
                                   if d.is_cable_playback else ""))
            except Exception as e:
                self.root.after(0, self.log, f"Beep failed on '{name}': {e}{_hint(str(e))}", logging.ERROR)

        threading.Thread(target=run, daemon=True).start()

    def _make_processor(self, c: AppConfig):
        if c.mode == "robot":
            return RobotVoiceProcessor(c.samplerate, c.robot_pitch_hz, c.ring_mod_hz, c.ring_mix)
        if c.mode == "voice":
            return make_voice(c.voice, c.samplerate, c.voice_pitch_st, c.voice_formant_st,
                              eq_amount=c.voice_eq_amount)
        if c.mode == "passthrough":
            return PassthroughProcessor()
        raise RuntimeError("AI voice mode is not available in this build.")

    def toggle_run(self):
        if self.engine and self.engine.stats.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.sample_page.recording:
            self.log("Finish the sample recording on 'Voice from sample' first.", logging.WARNING)
            return
        if self.voice_test_recording:
            self.log("Finish 'Test my voice' first (it is using the microphone).", logging.WARNING)
            return
        self.stop_voice_test_playback()
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
        self.run_btn.configure(text="■  Stop")
        F.set_enabled(self.rec_btn, True)
        self.set_state("Processing", proc.name)
        self.log(f"Started: {ind.name} -> {proc.name} -> {outd.name}" + (f" (+ monitor {mon.name})" if mon else ""))

    def stop(self):
        if self.engine:
            self.stop_record()
            s = self.engine.stats
            self.engine.stop()
            self.log(f"Stopped. blocks={s.blocks_out} dropped={s.dropped_blocks} underruns={s.underruns} "
                     f"delay fixes={s.catchups} ({s.catchup_ms:.0f} ms) errors={s.processing_errors} "
                     f"avg processing={s.proc_ms_avg:.1f} ms")
        self.run_btn.configure(text="▶  Start")
        F.set_enabled(self.rec_btn, False)
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
        self.rec_btn.configure(text="■  Stop recording")
        self.log(f"Recording processed voice to {path}")

    def stop_record(self):
        rec = self.engine.recorder if self.engine else None
        if rec is None:
            return
        self.engine.recorder = None
        secs = rec.close()
        self.rec_btn.configure(text="●  Record")
        extra = f", {rec.dropped} blocks skipped (disk too slow)" if rec.dropped else ""
        if rec.error:
            self.log(f"Recording failed: {rec.error}", logging.ERROR)
        else:
            self.log(f"Recording saved: {rec.path} ({secs:.1f} s{extra})")

    # ---- test my voice: record from the mic, play back with the selected voice (memory only, never to the cable)
    @property
    def voice_test_recording(self) -> bool:
        return self._test_rec is not None

    def _test_status(self, text, level=None):
        self.test_row.set_text(desc=text)
        if level is not None:
            self.log(text, level)

    def preview_output(self):
        """(index or None for the host API default, name) to play previews on, never the virtual cable."""
        import sounddevice as sd

        mon = self.mon_var.get()
        if mon and mon != MONITOR_OFF:
            d = dv.find_device(self.devices, mon, self.cfg.host_api, "output")
            if d is not None:
                return d.index, d.name
        for api in sd.query_hostapis():
            if api["name"] == self.cfg.host_api and api.get("default_output_device", -1) >= 0:
                idx = api["default_output_device"]
                name = sd.query_devices(idx)["name"]
                if any(h in name.lower() for h in dv.VIRTUAL_HINTS):
                    raise RuntimeError(f"the Windows default output is '{name}' (virtual cable); "
                                       "choose your headphones as Monitor on 'Audio devices'")
                return idx, name
        raise RuntimeError("no output device found; choose a Monitor device on 'Audio devices'")

    def toggle_voice_test(self):
        if self._test_rec is not None:
            self._finish_voice_test()
            return
        if self._test_busy:
            return
        self.stop_voice_test_playback()
        if self.engine and self.engine.stats.running:
            self._test_status("Press Stop first: the microphone is in use. While running, choose your headphones "
                              "as Monitor to hear yourself live.", logging.WARNING)
            return
        if self.sample_page.recording:
            self._test_status("Finish the sample recording on 'Voice from sample' first.", logging.WARNING)
            return
        mic = dv.find_device(self.devices, self.in_var.get(), self.cfg.host_api, "input")
        if mic is None:
            self._test_status("Choose your microphone on 'Audio devices' first.", logging.WARNING)
            return
        try:
            self.preview_output()   # fail now, not after the user has spoken
            self._test_rec = ClipRecorder(mic.index, self.cfg.host_api, self.cfg.samplerate, TEST_SECONDS)
        except Exception as e:
            self._test_status(f"Test failed: {e}{_hint(str(e))}", logging.WARNING)
            return
        F.set_enabled(self.replay_btn, False)
        self._test_status(f"Speak now... ({mic.name})")
        self._poll_voice_test()

    def _poll_voice_test(self):
        rec = self._test_rec
        if rec is None:
            return
        if rec.full:
            self._finish_voice_test()
            return
        left = max(0, TEST_SECONDS - int(rec.seconds))
        self.test_btn.configure(text=f"■  Stop ({left} s)")
        self.root.after(200, self._poll_voice_test)

    def _finish_voice_test(self):
        rec, self._test_rec = self._test_rec, None
        self.test_btn.configure(text=f"●  Speak {TEST_SECONDS} s")
        try:
            x = rec.stop()
        except Exception as e:
            self._test_status(f"Recording failed: {e}", logging.WARNING)
            return
        if x.size < 0.5 * self.cfg.samplerate:
            self._test_status("Too short – speak for at least half a second.")
            return
        if float(np.max(np.abs(x))) < 1e-3:
            self._test_status("Nothing was heard from the microphone. Check the microphone on 'Audio devices' "
                              "and Windows Privacy > Microphone access.", logging.WARNING)
            return
        self._test_take = x
        self.replay_voice_test()

    def replay_voice_test(self):
        """Render the last take with the voice and settings selected now, then play it (in a worker thread)."""
        if self._test_take is None or self._test_busy or self._test_rec is not None:
            return
        self.stop_voice_test_playback()
        c = self._collect_cfg()
        try:
            proc = self._make_processor(c)
            proc.reset()
            idx, name = self.preview_output()
        except Exception as e:
            self._test_status(f"Test failed: {e}", logging.WARNING)
            return
        take, sr, api = self._test_take, c.samplerate, c.host_api
        block = int(round(c.dsp_block_time * sr))
        self._test_busy = True
        F.set_enabled(self.test_btn, False)
        F.set_enabled(self.replay_btn, False)
        self._test_status(f"Preparing '{proc.name}'...")

        def run():   # rendering only; the stream is opened on the UI thread (WASAPI needs COM there)
            try:
                y, err = render(proc, take, sr, block, c.input_gate_db, c.output_gain_db), None
            except Exception as e:
                y, err = None, e
            self.root.after(0, self._voice_test_play, y, err, proc.name, idx, name)

        threading.Thread(target=run, daemon=True).start()

    def _voice_test_play(self, y, err, voice, idx, device):
        self._test_busy = False
        F.set_enabled(self.test_btn, True)
        F.set_enabled(self.replay_btn, True)
        if err is None and (self._test_rec is not None or (self.engine and self.engine.stats.running)):
            return                   # the user started something else meanwhile
        try:
            if err is not None:
                raise err
            self._test_player = ClipPlayer(idx, self.cfg.host_api, y, self.cfg.samplerate)
        except Exception as e:
            self._test_status(f"Playback failed: {e}{_hint(str(e))}", logging.WARNING)
            return
        self._test_status(f"Playing {y.size / self.cfg.samplerate:.1f} s as '{voice}' on '{device}' – this is how "
                          "Teams will hear you. Pick another voice and press ▶ Play again to compare.")
        self.log(f"Test my voice: played as '{voice}' on '{device}'.")

    def stop_voice_test_playback(self):
        if self._test_player is not None:
            try:
                self._test_player.stop()
            except Exception:
                pass
            self._test_player = None

    def _on_fatal(self, msg):  # called from engine threads
        self._fatal_msg = msg

    def run_diagnostics(self, verify_gpu=False):
        self.diag.delete("1.0", "end")
        self.diag.insert("end", "Running" + (" GPU inference test" if verify_gpu else "") + "...\n")
        for b in (self.diag_btn, self.diag_gpu_btn):
            F.set_enabled(b, False)

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
        F.set_enabled(self.diag_btn, True)
        if self.ai_available:
            F.set_enabled(self.diag_gpu_btn, True)

    # ================================================================ periodic UI update
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

                sys_txt = (f"CPU {psutil.cpu_percent(None):.0f}% (app {self.proc.cpu_percent(None) / psutil.cpu_count():.0f}%)"
                           f"   ·   RAM {self.proc.memory_info().rss / 2**20:.0f} MB")
            except Exception:
                pass
        if self.engine and self.engine.stats.running:
            s = self.engine.stats
            self.in_meter.set_db(s.input_db)
            self.out_meter.set_db(s.output_db)
            rec = self.engine.recorder
            if rec is not None:
                self.rec_btn.configure(text=f"■  Stop recording ({rec.seconds:.0f} s)")
            dev = s.extra.get("device_latency_ms") or {}
            dev_txt = f" + device (in {dev['input']:.0f} / out {dev['output']:.0f} ms)" if dev else ""
            text = (f"Processing {s.proc_ms_avg:.1f} ms (max {s.proc_ms_max:.1f})   ·   RTF {s.rtf:.2f}   ·   "
                    f"Latency ~{s.est_latency_ms:.0f} ms{dev_txt}\n"
                    f"Dropped {s.dropped_blocks}   ·   Underruns {s.underruns}   ·   Delay fixes {s.catchups}"
                    f"   ·   Errors {s.processing_errors}"
                    f"   ·   {sys_txt}")
        else:
            self.in_meter.set_db(-120)
            self.out_meter.set_db(-120)
            text = sys_txt
        if text != self._stats_text:
            self._stats_text = text
            self.stats_lbl.configure(text=text)
        self.root.after(200, self._poll)

    def on_close(self):
        try:
            self.sample_page.close()
            self.stop_voice_test_playback()
            if self._test_rec is not None:
                self._test_rec.stop()
                self._test_rec = None
            if self.engine:
                self.engine.stop()
            self._collect_cfg().save(paths.CONFIG_FILE)
        except Exception:
            logger.exception("Error during shutdown")
        self.root.destroy()


def main():
    from logging.handlers import RotatingFileHandler

    paths.ensure_dirs()
    if sys.stdout is None or sys.stderr is None:    # started without a console (pythonw / Parroty.exe)
        devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stdout or devnull
        sys.stderr = sys.stderr or devnull
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[RotatingFileHandler(paths.LOGS_DIR / "rtvc.log", maxBytes=2_000_000,
                                                      backupCount=3, encoding="utf-8"), logging.StreamHandler()])
    logger.info("Starting Parroty %s", __version__)
    try:    # own taskbar entry with the Parroty icon instead of being grouped under python.exe
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Parroty.VoiceChanger")
    except Exception:
        pass
    root = ctk.CTk()           # CustomTkinter enables per-monitor DPI awareness itself
    try:
        App(root)
    except Exception as e:
        logger.exception("UI failed to start")
        messagebox.showerror("Parroty", f"Failed to start: {e}")
        raise
    root.mainloop()

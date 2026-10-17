"""Windows 11 (Fluent) look for the CustomTkinter UI.

Colours are (light, dark) pairs, so the window follows the Windows light/dark setting.
Values follow the WinUI 3 defaults (accent = Windows blue #005FB8 / #60CDFF).
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont

import customtkinter as ctk

# ---------------------------------------------------------------- colours (light, dark)
ACCENT = ("#005FB8", "#60CDFF")
ACCENT_HOVER = ("#196EBF", "#5AB8E8")
ON_ACCENT = ("#FFFFFF", "#000000")
ACCENT_SOFT = ("#E6F0FA", "#2A3946")        # selected tile background
WINDOW = ("#F3F3F3", "#202020")
CARD = ("#FBFBFB", "#2B2B2B")
CARD_HOVER = ("#F5F5F5", "#323232")
BORDER = ("#E5E5E5", "#1D1D1D")
NAV_HOVER = ("#EAEAEA", "#2D2D2D")
NAV_SELECTED = ("#E4E4E4", "#2F2F2F")
CONTROL = ("#FFFFFF", "#373737")
CONTROL_HOVER = ("#F6F6F6", "#3D3D3D")
CONTROL_BORDER = ("#D1D1D1", "#474747")
CONTROL_DISABLED = ("#E0E0E0", "#3A3A3A")
TRACK = ("#C8C8C8", "#5C5C5C")
TEXT = ("#1A1A1A", "#FFFFFF")
TEXT2 = ("#5D5D5D", "#C5C5C5")
TEXT_DISABLED = ("#A0A0A0", "#7A7A7A")
SUCCESS = ("#0F7B0F", "#6CCB5F")
CAUTION = ("#9D5D00", "#FCE100")
CRITICAL = ("#C42B1C", "#FF99A4")
INFO_BG = ("#EAF2FB", "#26313B")
CAUTION_BG = ("#FFF4CE", "#433519")

# ---------------------------------------------------------------- icons (Segoe Fluent Icons code points)
I_MIC, I_SPEAKERS, I_MESSAGE, I_PULSE, I_HELP = "", "", "", "", ""
I_THEME, I_INFO, I_ROBOT, I_EMOJI, I_MUSIC = "", "", "", "", ""
I_RADIO, I_VOLUME, I_HEADPHONES, I_SETTINGS = "", "", "", ""

# voice key -> (icon, uses the symbol font instead of the icon font)
VOICE_ICONS = {
    "robot": (I_ROBOT, False), "passthrough": (I_MIC, False),
    "deep_male": ("♂", True), "big_man": ("♂", True), "young_male": ("♂", True),
    "female": ("♀", True), "soft_female": ("♀", True), "child": (I_EMOJI, False),
    "chipmunk": (I_MUSIC, False), "monster": ("☠", True), "radio": (I_RADIO, False),
    "whisper": ("☁", True), "cave": ("⛰", True), "alien": ("👽", True),
}


class Fonts:
    """Segoe UI Variable on Windows 11, Segoe UI on Windows 10. Sizes are CustomTkinter px (DPI-scaled)."""

    def __init__(self):
        fam = set(tkfont.families())

        def pick(*names):
            return next((n for n in names if n in fam), "Segoe UI")

        text = pick("Segoe UI Variable Text", "Segoe UI")
        strong = pick("Segoe UI Variable Text Semibold", "Segoe UI Semibold")
        display = pick("Segoe UI Variable Display Semib", "Segoe UI Semibold")
        self.icons = pick("Segoe Fluent Icons", "Segoe MDL2 Assets", "Segoe UI Symbol")
        self.body = (text, 14)
        self.body_strong = (strong, 14)
        self.caption = (text, 12)
        self.subtitle = (display, 20)
        self.title = (display, 28)
        self.mono = ("Cascadia Mono" if "Cascadia Mono" in fam else "Consolas", 12)

    def icon(self, size=16, symbol=False):
        return ("Segoe UI Symbol" if symbol else self.icons, size)


# ---------------------------------------------------------------- small helpers
def _inside(widget) -> bool:
    w = widget.winfo_containing(*widget.winfo_pointerxy())
    while w is not None:
        if w == widget:
            return True
        w = w.master
    return False


def clickable(frame, parts, on_click, on_hover):
    """Make a frame and its labels behave like one button (click + hover)."""
    def leave(_):
        frame.after(1, lambda: frame.winfo_exists() and on_hover(_inside(frame)))

    for w in (frame, *parts):
        w.bind("<Button-1>", lambda _e: on_click())
        w.bind("<Enter>", lambda _e: on_hover(True))
        w.bind("<Leave>", leave)
        try:
            w.configure(cursor="hand2")
        except (tk.TclError, ValueError):
            pass


def button(parent, text, command, fonts: Fonts, accent=False, width=0, **kw):
    b = ctk.CTkButton(parent, text=text, command=command, width=width or 120, height=32, corner_radius=5,
                      font=fonts.body, fg_color=ACCENT if accent else CONTROL,
                      hover_color=ACCENT_HOVER if accent else CONTROL_HOVER,
                      text_color=ON_ACCENT if accent else TEXT, text_color_disabled=TEXT_DISABLED,
                      border_width=0 if accent else 1, border_color=CONTROL_BORDER, **kw)
    b.is_accent = accent
    return b


def link(parent, text, command, fonts: Fonts):
    return ctk.CTkButton(parent, text=text, command=command, width=0, height=24, corner_radius=4,
                         font=fonts.body, fg_color="transparent", hover_color=NAV_HOVER, text_color=ACCENT)


def set_enabled(widget, on: bool):
    widget.configure(state="normal" if on else "disabled")
    if getattr(widget, "is_accent", False):
        widget.configure(fg_color=ACCENT if on else CONTROL_DISABLED)
    if isinstance(widget, ctk.CTkSlider):
        widget.configure(button_color=ACCENT if on else TEXT_DISABLED,
                         progress_color=ACCENT if on else TEXT_DISABLED)


def card(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=CARD, border_color=BORDER, border_width=1, corner_radius=7, **kw)


def label(parent, text, font, color=TEXT, **kw):
    return ctk.CTkLabel(parent, text=text, font=font, text_color=color, **kw)


def option_menu(parent, variable, command, fonts: Fonts, width=340):
    """A drop-down with a 1 px border, like a WinUI ComboBox."""
    box = ctk.CTkFrame(parent, fg_color=CONTROL, border_color=CONTROL_BORDER, border_width=1, corner_radius=5)
    om = ctk.CTkOptionMenu(box, variable=variable, values=[], command=command, width=width, height=30,
                           corner_radius=4, dynamic_resizing=False, anchor="w", font=fonts.body,
                           dropdown_font=fonts.body, fg_color=CONTROL, button_color=CONTROL,
                           button_hover_color=CONTROL_HOVER, text_color=TEXT, text_color_disabled=TEXT_DISABLED,
                           dropdown_fg_color=CARD, dropdown_hover_color=NAV_HOVER, dropdown_text_color=TEXT)
    om.pack(padx=1, pady=1)
    return box, om


class SettingRow(ctk.CTkFrame):
    """A Windows Settings style row: icon, title + description on the left, controls on the right."""

    def __init__(self, parent, fonts: Fonts, icon, title, desc="", symbol=False):
        super().__init__(parent, fg_color=CARD, border_color=BORDER, border_width=1, corner_radius=7)
        self.columnconfigure(1, weight=1)
        self.icon = label(self, icon, fonts.icon(18, symbol), width=24)
        self.icon.grid(row=0, column=0, rowspan=2, padx=(18, 14), pady=14)
        self.title = label(self, title, fonts.body, anchor="w", justify="left")
        self.title.grid(row=0, column=1, sticky="sw", pady=(12, 0) if desc else 14)
        self.desc = label(self, desc, fonts.caption, TEXT2, anchor="w", justify="left", wraplength=420)
        if desc:
            self.desc.grid(row=1, column=1, sticky="nw", pady=(0, 12))
        self.right = ctk.CTkFrame(self, fg_color="transparent")
        self.right.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 16))
        self.bind("<Configure>", self._rewrap, add="+")

    def _rewrap(self, _event=None):
        """Wrap the description to the space left of the controls, so it is never clipped."""
        scale = self._get_widget_scaling()
        free = (self.winfo_width() - self.right.winfo_reqwidth()) / scale - 24 - 32 - 12 - 16 - 8
        wrap = int(max(160, min(560, free)))
        if wrap != self.desc.cget("wraplength"):
            self.desc.configure(wraplength=wrap)

    def set_text(self, title=None, desc=None, icon=None, fonts: Fonts | None = None, symbol=False):
        if title is not None:
            self.title.configure(text=title)
        if desc is not None:
            self.desc.configure(text=desc)
        if icon is not None and fonts is not None:
            self.icon.configure(text=icon, font=fonts.icon(18, symbol))


def slider(parent, fonts: Fonts, lo, hi, value, digits, steps, unit=""):
    """Slider + value text; returns (DoubleVar, CTkSlider)."""
    var = tk.DoubleVar(value=value)
    s = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=steps, variable=var, width=220, height=18,
                      fg_color=TRACK, progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER)
    s.grid(row=0, column=0)
    val = label(parent, "", fonts.body, TEXT2, width=64, anchor="e")
    val.grid(row=0, column=1, padx=(8, 0))
    fmt = f"{{:+.{digits}f}}" if lo < 0 < hi else f"{{:.{digits}f}}"

    def show(*_):
        try:
            val.configure(text=fmt.format(var.get()) + unit)
        except tk.TclError:
            pass

    var.trace_add("write", show)
    show()
    return var, s


class Tile(ctk.CTkFrame):
    """Selectable voice card."""

    def __init__(self, parent, fonts: Fonts, icon, symbol, title, subtitle, command):
        super().__init__(parent, fg_color=CARD, border_color=BORDER, border_width=1, corner_radius=7, height=96)
        self._sel = self._hover = False
        self.icon = label(self, icon, fonts.icon(22, symbol), ACCENT)
        self.icon.pack(pady=(12, 2))
        self.title = label(self, title, fonts.body_strong, wraplength=96, justify="center")
        self.title.pack(padx=6)
        self.sub = label(self, subtitle, fonts.caption, TEXT2, height=18)
        self.sub.pack(padx=6, pady=(0, 10))
        clickable(self, (self.icon, self.title, self.sub), command, self._on_hover)

    def _on_hover(self, h):
        self._hover = h
        self._paint()

    def set_selected(self, sel: bool):
        self._sel = sel
        self._paint()

    def _paint(self):
        self.configure(fg_color=ACCENT_SOFT if self._sel else CARD_HOVER if self._hover else CARD,
                       border_color=ACCENT if self._sel else CONTROL_BORDER if self._hover else BORDER,
                       border_width=2 if self._sel else 1)


class NavItem(ctk.CTkFrame):
    """Left navigation pane item with the WinUI selection pill."""

    def __init__(self, parent, fonts: Fonts, icon, text, command):
        super().__init__(parent, fg_color="transparent", corner_radius=5, height=38)
        self._sel = self._hover = False
        self.pill = ctk.CTkFrame(self, width=3, height=16, corner_radius=2, fg_color="transparent")
        self.pill.grid(row=0, column=0, padx=(2, 0))
        self.icon = label(self, icon, fonts.icon(16), width=20)
        self.icon.grid(row=0, column=1, padx=(10, 12), pady=8)
        self.text = label(self, text, fonts.body, anchor="w")
        self.text.grid(row=0, column=2, sticky="w", padx=(0, 12))
        self.columnconfigure(2, weight=1)
        clickable(self, (self.pill, self.icon, self.text), command, self._on_hover)

    def _on_hover(self, h):
        self._hover = h
        self._paint()

    def set_selected(self, sel: bool):
        self._sel = sel
        self._paint()

    def _paint(self):
        self.configure(fg_color=NAV_SELECTED if self._sel else NAV_HOVER if self._hover else "transparent")
        self.pill.configure(fg_color=ACCENT if self._sel else "transparent")


class Meter(ctk.CTkFrame):
    """Level bar in dBFS (real values from the engine)."""

    FLOOR = -60.0

    def __init__(self, parent, fonts: Fonts, text):
        super().__init__(parent, fg_color="transparent")
        self.columnconfigure(1, weight=1)
        label(self, text, fonts.caption, TEXT2, width=52, anchor="w").grid(row=0, column=0)
        self.bar = ctk.CTkProgressBar(self, height=6, corner_radius=3, fg_color=TRACK, progress_color=SUCCESS)
        self.bar.grid(row=0, column=1, sticky="ew", padx=(4, 10))
        self.bar.set(0)
        self.val = label(self, "–", fonts.caption, TEXT2, width=70, anchor="e")
        self.val.grid(row=0, column=2)
        self._last = None

    def set_db(self, db: float):
        key = round(db) if db > -120 else None
        if key == self._last:
            return
        self._last = key
        frac = min(max((db - self.FLOOR) / -self.FLOOR, 0.0), 1.0)
        self.bar.set(frac)
        self.bar.configure(progress_color=SUCCESS if db < -12 else CAUTION if db < -3 else CRITICAL)
        self.val.configure(text=f"{db:.0f} dBFS" if key is not None else "–")

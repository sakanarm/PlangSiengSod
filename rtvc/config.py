"""Application settings: defaults, validation and JSON persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "onnx-dml", "onnx-cpu", "torch-cpu")
F0_METHODS = ("rmvpe", "pm")
SAMPLE_RATES = (44100, 48000)
# robot / voice = DSP effects (no model), passthrough = routing test, rvc = AI conversion (optional add-on)
MODES = ("robot", "voice", "passthrough", "rvc")

# Hard limits. The pitch cache in the converter holds 1024 frames of 10 ms, so the
# whole analysis window (extra + crossfade + block) must stay well below 10.24 s.
LIMITS = {
    "block_time": (0.05, 1.5),
    "crossfade_time": (0.01, 0.15),
    "extra_time": (0.1, 5.0),
    "pitch_shift": (-24, 24),
    "index_rate": (0.0, 1.0),
    "rms_mix_rate": (0.0, 1.0),
    "input_gate_db": (-60.0, 0.0),
    "output_gain_db": (-24.0, 12.0),
    "output_prebuffer_blocks": (0, 4),
    "dml_device_id": (0, 15),
    "dsp_block_time": (0.01, 0.2),
    "robot_pitch_hz": (40.0, 400.0),
    "ring_mod_hz": (0.0, 200.0),
    "ring_mix": (0.0, 1.0),
    "voice_pitch_st": (-12.0, 12.0),
    "voice_formant_st": (-8.0, 8.0),
}


@dataclass
class AppConfig:
    # Devices are stored by name (indices change when devices are plugged in/out).
    host_api: str = "Windows WASAPI"
    input_device: str = ""
    output_device: str = ""           # normally "CABLE Input (VB-Audio Virtual Cable)"
    monitor_device: str = ""          # optional: headphones to hear yourself ("" = off)
    samplerate: int = 48000

    mode: str = "robot"
    dsp_block_time: float = 0.05      # block size for robot / pass-through (low latency)
    robot_pitch_hz: float = 110.0     # monotone pitch of the robot voice
    ring_mod_hz: float = 30.0         # 0 = off; adds a metallic wobble
    ring_mix: float = 0.3
    voice: str = "deep_male"          # key in rtvc.audio.voices.VOICES (DSP voices, no model)
    voice_pitch_st: float = -4.0      # semitones, adjustable per voice
    voice_formant_st: float = -2.0    # semitones, + = smaller / younger sounding speaker

    model_path: str = ""
    index_path: str = ""
    backend: str = "auto"
    dml_device_id: int = 0

    pitch_shift: int = 0              # semitones
    f0_method: str = "pm"             # "rmvpe" is more robust but needs a 345 MB download
    index_rate: float = 0.5
    rms_mix_rate: float = 0.25        # 1.0 = keep model loudness, 0.0 = follow mic loudness
    input_gate_db: float = -60.0      # -60 disables the gate
    output_gain_db: float = 0.0

    block_time: float = 0.30          # seconds of audio converted per step
    crossfade_time: float = 0.05
    extra_time: float = 1.0           # left context fed to the model for quality
    output_prebuffer_blocks: int = 1  # jitter buffer in blocks

    consent_confirmed: bool = False
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list means valid."""
        errors: list[str] = []
        for name, (lo, hi) in LIMITS.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{name} must be a number, got {value!r}")
            elif not lo <= value <= hi:
                errors.append(f"{name}={value} is outside the allowed range {lo}..{hi}")
        if self.backend not in BACKENDS:
            errors.append(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.mode not in MODES:
            errors.append(f"mode must be one of {MODES}, got {self.mode!r}")
        from .audio.voices import VOICES

        if self.voice not in VOICES:
            errors.append(f"voice must be one of {tuple(VOICES)}, got {self.voice!r}")
        if self.f0_method not in F0_METHODS:
            errors.append(f"f0_method must be one of {F0_METHODS}, got {self.f0_method!r}")
        if self.samplerate not in SAMPLE_RATES:
            errors.append(f"samplerate must be one of {SAMPLE_RATES}, got {self.samplerate!r}")
        if isinstance(self.crossfade_time, (int, float)) and isinstance(self.block_time, (int, float)):
            if self.crossfade_time >= self.block_time:
                errors.append("crossfade_time must be shorter than block_time")
        try:
            window = self.extra_time + self.block_time + self.crossfade_time + 0.01
            if window > 9.5:
                errors.append(
                    f"extra_time + block_time + crossfade_time = {window:.2f}s exceeds 9.5s"
                )
        except TypeError:
            pass
        if self.output_device and self.input_device and self.output_device == self.input_device:
            errors.append("Input and output device must be different")
        if self.monitor_device and self.monitor_device == self.output_device:
            errors.append("Monitor device must differ from the output (virtual cable) device")
        return errors

    # ----------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                logger.warning("Ignoring unknown config key %r", key)
                continue
            kwargs[key] = value
        cfg = cls(**kwargs)
        # Coerce ints that JSON may have turned into floats (and vice versa).
        for name in ("pitch_shift", "output_prebuffer_blocks", "dml_device_id", "samplerate"):
            v = getattr(cfg, name)
            if isinstance(v, float) and v.is_integer():
                setattr(cfg, name, int(v))
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load settings; on a missing or corrupt file return defaults (and log why)."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top-level JSON value is not an object")
            return cls.from_dict(data)
        except (ValueError, TypeError) as exc:
            logger.error("Config file %s is invalid (%s); using defaults", path, exc)
            return cls()
